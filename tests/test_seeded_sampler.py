"""Tests for per-identifier seeding of the decoder's sampler.

The rescue pass decodes from temperature 0.2, and above 0.0 faster-whisper samples
rather than beam-searches, so until 0.6.2 the rescue was an unrecorded draw. Selection
differed on two of eleven recordings between two runs of the same build, and a 67 word
swing between two draws of the same rescue flipped what got published on tcf.20240713.

0.6.2 seeds every decode from a hash of the job GUID. The properties that matter:

  1. the same GUID gives the same seed, in this process and in any other,
  2. different GUIDs give different seeds, so the archive keeps its variety,
  3. `RESCUE_SEED_MODE=off` sets nothing, restoring the 0.6.1 behaviour,
  4. the derivation is pinned, because changing it silently would change every
     transcript the service produces from then on,
  5. the worker has no concurrent decode path, which is what makes a process-global
     seed call safe.

The real module is loaded by the shared `transcribe_module` fixture in conftest.py with
the heavy GPU imports stubbed. No GPU, no model, no audio.
"""

import ast
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------------------
# The derivation
# --------------------------------------------------------------------------------------
def test_the_same_guid_gives_the_same_seed(transcribe_module):
    guid = "f5e593e9-c7b8-4809-a476-2a82d7b613f3"
    assert transcribe_module.seed_for_guid(guid) == transcribe_module.seed_for_guid(guid)


@pytest.mark.parametrize(
    "left,right",
    [
        ("f5e593e9-c7b8-4809-a476-2a82d7b613f3", "00000000-0000-0000-0000-000000000000"),
        ("tcf.20240713", "tcf.20241105"),
        ("a", "b"),
    ],
)
def test_different_guids_give_different_seeds(transcribe_module, left, right):
    """A constant seed would fix every recording to one corner of the sampler. The
    point of hashing the GUID is that the archive keeps the spread it has today."""
    assert transcribe_module.seed_for_guid(left) != transcribe_module.seed_for_guid(right)


def test_the_derivation_is_pinned(transcribe_module):
    """Changing this changes every transcript decoded from here on, so it must not be
    possible to change it by accident during a refactor."""
    assert transcribe_module.seed_for_guid(
        "f5e593e9-c7b8-4809-a476-2a82d7b613f3"
    ) == 2700934071
    assert transcribe_module.seed_for_guid(
        "00000000-0000-0000-0000-000000000000"
    ) == 936449225
    assert transcribe_module.seed_for_guid("abc-123") == 3985580431


def test_the_seed_fits_what_ctranslate2_accepts(transcribe_module):
    """`set_random_seed` takes an unsigned 32 bit int, and CTranslate2 reads the largest
    such value as "no seed was set", so the derivation must never produce it."""
    for guid in [f"guid-{n}" for n in range(500)]:
        seed = transcribe_module.seed_for_guid(guid)
        assert isinstance(seed, int)
        assert 0 <= seed < 2**32 - 1


def test_a_missing_guid_still_yields_a_seed(transcribe_module):
    """Nothing should decode without a GUID, but a seeding helper must not be the thing
    that raises if one ever does."""
    assert isinstance(transcribe_module.seed_for_guid(None), int)
    assert isinstance(transcribe_module.seed_for_guid(""), int)


# --------------------------------------------------------------------------------------
# Across processes
# --------------------------------------------------------------------------------------
_SEED_IN_SUBPROCESS = """
import importlib.util
import sys
import types

fake_torch = types.ModuleType("torch")
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
fake_fw = types.ModuleType("faster_whisper")
fake_fw.WhisperModel = object
fake_pydub = types.ModuleType("pydub")
fake_pydub.AudioSegment = object
fake_utils = types.ModuleType("pydub.utils")
fake_utils.mediainfo = lambda path: {}
fake_pydub.utils = fake_utils
sys.modules["torch"] = fake_torch
sys.modules["faster_whisper"] = fake_fw
sys.modules["pydub"] = fake_pydub
sys.modules["pydub.utils"] = fake_utils

repo_root, guid = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo_root)
spec = importlib.util.spec_from_file_location(
    "transcribe_in_subprocess", repo_root + "/transcribe.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.RESCUE_SEED_MODE, module.seed_for_guid(guid))
"""


def _mode_and_seed_in_subprocess(guid, hash_seed="0", mode=None):
    """Import transcribe.py in a fresh interpreter and report the mode it resolved and
    the seed it derives. The module reads the environment once, at import, so this is
    the only way to exercise what a container actually starts with."""
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    env.pop("RESCUE_SEED_MODE", None)
    if mode is not None:
        env["RESCUE_SEED_MODE"] = mode
    result = subprocess.run(
        [sys.executable, "-c", _SEED_IN_SUBPROCESS, REPO_ROOT, guid],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    resolved, seed = result.stdout.split()
    return resolved, None if seed == "None" else int(seed)


def _seed_in_subprocess(guid, hash_seed):
    return _mode_and_seed_in_subprocess(guid, hash_seed)[1]


def test_the_seed_is_stable_across_processes(transcribe_module):
    """Python salts `hash()` per process. Two interpreters started with different hash
    seeds must still derive the same decoder seed for the same recording, or a restart
    silently changes what a re-decode produces."""
    guid = "f5e593e9-c7b8-4809-a476-2a82d7b613f3"
    first = _seed_in_subprocess(guid, "0")
    second = _seed_in_subprocess(guid, "1")
    assert first == second
    assert first == transcribe_module.seed_for_guid(guid)


# --------------------------------------------------------------------------------------
# The switch
# --------------------------------------------------------------------------------------
def test_seeding_is_on_by_default(transcribe_module):
    assert transcribe_module.RESCUE_SEED_MODE == "guid"


def test_off_derives_no_seed(transcribe_module, monkeypatch):
    monkeypatch.setattr(transcribe_module, "RESCUE_SEED_MODE", "off")
    assert transcribe_module.seed_for_guid("f5e593e9") is None


def test_off_sets_no_seed(transcribe_module, monkeypatch):
    """The point of the switch is that the previous behaviour is restorable on a
    running container, so `off` must reach ctranslate2 not at all."""
    monkeypatch.setattr(transcribe_module, "RESCUE_SEED_MODE", "off")
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        type(sys)("ctranslate2"),
    )
    sys.modules["ctranslate2"].set_random_seed = calls.append
    assert transcribe_module.apply_decode_seed("f5e593e9") is None
    assert calls == []


def test_guid_mode_sets_the_derived_seed(transcribe_module, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "ctranslate2", type(sys)("ctranslate2"))
    sys.modules["ctranslate2"].set_random_seed = calls.append
    seed = transcribe_module.apply_decode_seed("abc-123")
    assert seed == 3985580431
    assert calls == [3985580431]


def test_a_seeding_failure_does_not_fail_the_job(transcribe_module, monkeypatch):
    """An unseeded transcript is what every release up to 0.6.1 produced. A refused job
    is worse."""
    def boom(_seed):
        raise RuntimeError("no ctranslate2 here")

    monkeypatch.setitem(sys.modules, "ctranslate2", type(sys)("ctranslate2"))
    sys.modules["ctranslate2"].set_random_seed = boom
    assert transcribe_module.apply_decode_seed("abc-123") is None


@pytest.mark.parametrize("mode", ["GUID", " guid ", "", None])
def test_these_all_mean_seeded(mode):
    """The default, and the spellings of it a deployment is likely to produce."""
    resolved, seed = _mode_and_seed_in_subprocess("abc-123", mode=mode)
    assert resolved == "guid"
    assert seed == 3985580431


@pytest.mark.parametrize("mode", ["off", "OFF"])
def test_off_survives_the_import(mode):
    """Withdrawing seeding has to work from the container's environment, not just from
    a monkeypatched attribute."""
    resolved, seed = _mode_and_seed_in_subprocess("abc-123", mode=mode)
    assert resolved == "off"
    assert seed is None


def test_an_unknown_mode_falls_back_to_seeding(transcribe_module):
    """A typo in the deployment must not silently return the service to unrecorded
    draws. Whatever the variable says, the module only ever runs one of the two modes."""
    resolved, seed = _mode_and_seed_in_subprocess("abc-123", mode="sometimes")
    assert resolved == "guid"
    assert seed == 3985580431
    assert transcribe_module.RESCUE_SEED_MODE in transcribe_module._SEED_MODES


# --------------------------------------------------------------------------------------
# Why a process-global seed is safe: one worker, one decode at a time
# --------------------------------------------------------------------------------------
def _app_source_tree():
    with open(os.path.join(REPO_ROOT, "app.py")) as handle:
        return ast.parse(handle.read())


def _thread_constructions(tree):
    """Every `threading.Thread(...)` construction in app.py, with its target name."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "Thread":
            continue
        target = ""
        for keyword in node.keywords:
            if keyword.arg == "target":
                target = getattr(keyword.value, "id", "") or getattr(
                    keyword.value, "attr", ""
                )
        found.append(target)
    return found


def test_the_worker_has_no_concurrent_decode_path():
    """`ctranslate2.set_random_seed` is process-global: it seeds the generator, not a
    call. That is only safe while exactly one decode runs at a time in this process.

    app.py starts one `transcription-worker` thread and no other. If a second decoding
    thread or a pool is ever added, two jobs interleave, each one reseeds the generator
    the other is drawing from, and reproducibility is gone with no visible symptom. This
    test is the tripwire on that assumption: if it fails, seeding needs to move behind a
    lock held across the decode, or the seed needs to become per-call.
    """
    targets = _thread_constructions(_app_source_tree())
    assert targets == ["transcription_worker"], (
        "app.py constructs threads other than the single transcription worker; "
        "a process-global decoder seed is no longer safe"
    )


def test_the_worker_thread_is_started_only_once(app_module, monkeypatch):
    """A second live worker would be a second decode path even with one Thread call."""
    class _AliveThread:
        def is_alive(self):
            return True

    existing = _AliveThread()
    monkeypatch.setattr(app_module, "worker_thread", existing)
    assert app_module.start_worker() is existing


def test_the_model_is_loaded_with_one_worker(transcribe_module):
    """faster-whisper's `num_workers` above 1 permits concurrent `transcribe` calls on
    the same model, which is the other way two decodes could overlap."""
    assert transcribe_module.WHISPER_NUM_WORKERS == 1


# --------------------------------------------------------------------------------------
# What the change does, and does not, do to a decode
# --------------------------------------------------------------------------------------
class _Word:
    def __init__(self, start, end, word):
        self.start, self.end, self.word, self.probability = start, end, word, 0.9


class _Seg:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text
        self.avg_logprob, self.no_speech_prob, self.compression_ratio = -0.25, 0.01, 1.4
        self.temperature, self.id, self.seek = 0.0, 0, 0
        self.tokens = []
        tokens = text.split()
        step = (end - start) / max(len(tokens), 1)
        self.words = [
            _Word(start + i * step, start + (i + 1) * step, (" " if i else "") + t)
            for i, t in enumerate(tokens)
        ]


NORMAL_RATE_BODY = (
    "and so the word of the Lord came to him once again saying "
    "behold I will send my messenger before your face to prepare the way"
)


def _clean(n=20):
    return [_Seg(i * 10.0, i * 10.0 + 10.0, NORMAL_RATE_BODY) for i in range(n)]


def _anomalous(n=20):
    """The shape that fires the rescue: six segments with almost no words."""
    segments = _clean(n)
    for i in (8, 9, 10, 11, 12, 13):
        segments[i] = _Seg(i * 10.0, i * 10.0 + 10.0, "and then")
    return segments


@pytest.fixture
def scripted_decode(transcribe_module, monkeypatch, tmp_path):
    """Run transcribe_audio against a scripted model, logging seeds and decodes in the
    order they happen. Returns (events, result)."""
    def run(pass_segments, guid="f5e593e9-c7b8-4809-a476-2a82d7b613f3"):
        events = []

        stub = type(sys)("ctranslate2")
        stub.set_random_seed = lambda seed: events.append(("seed", seed))
        monkeypatch.setitem(sys.modules, "ctranslate2", stub)

        class _FakeModel:
            def transcribe(self, audio, **kwargs):
                events.append(("decode", kwargs))
                index = min(
                    len([e for e in events if e[0] == "decode"]) - 1,
                    len(pass_segments) - 1,
                )
                return (pass_segments[index], None)

        monkeypatch.setattr(transcribe_module, "load_whisper_model", lambda: _FakeModel())
        monkeypatch.setattr(transcribe_module, "get_audio_duration", lambda path: 200.0)
        monkeypatch.setattr(
            transcribe_module, "measure_mean_dbfs", lambda path, duration=0: -20.0
        )
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        return events, transcribe_module.transcribe_audio(str(audio), guid)

    return run


def test_every_decode_is_seeded_immediately_before_it_runs(scripted_decode):
    """Anything between the seed call and the decode could consume draws from the
    generator, so the order is part of the contract, not an implementation detail."""
    events, result = scripted_decode([_anomalous(), _clean()])
    assert result["rescue_attempted"] is True
    assert [kind for kind, _ in events] == ["seed", "decode", "seed", "decode"]


def test_both_passes_of_a_job_run_under_the_same_seed(scripted_decode):
    """One GUID, one draw. The rescue is not a second roll of the dice."""
    events, _ = scripted_decode([_anomalous(), _clean()])
    seeds = [value for kind, value in events if kind == "seed"]
    assert seeds == [2700934071, 2700934071]


def test_the_published_seed_is_reported(scripted_decode):
    events, result = scripted_decode([_clean()])
    assert result["seed"] == 2700934071


def test_seeding_off_reports_no_seed(scripted_decode, transcribe_module, monkeypatch):
    monkeypatch.setattr(transcribe_module, "RESCUE_SEED_MODE", "off")
    events, result = scripted_decode([_clean()])
    assert [kind for kind, _ in events] == ["decode"]
    assert result["seed"] is None


def test_the_seed_reaches_the_retained_review_record(
    scripted_decode, transcribe_module, monkeypatch, tmp_path
):
    """The retained pair exists so a discarded transcript can be reviewed later. Without
    the seed beside it, it still could not be reproduced."""
    import json

    kept = tmp_path / "kept"
    monkeypatch.setattr(transcribe_module, "RESCUE_TRANSCRIPT_DIR", str(kept))
    guid = "abc-123"
    scripted_decode([_anomalous(), _clean()], guid=guid)
    record = json.loads((kept / f"{guid}.json").read_text())
    assert [p["seed"] for p in record["passes"]] == [3985580431, 3985580431]


def test_the_primary_decode_is_unchanged_by_seeding(scripted_decode, transcribe_module,
                                                    monkeypatch):
    """Seeding must not have touched what reaches the beam search. The primary decodes
    from temperature 0.0, where faster-whisper beam-searches and does not sample, so the
    0.6.1 primary and the 0.6.2 primary have to be the same decode."""
    seeded_events, seeded = scripted_decode([_clean()])
    seeded_kwargs = [kwargs for kind, kwargs in seeded_events if kind == "decode"]

    monkeypatch.setattr(transcribe_module, "RESCUE_SEED_MODE", "off")
    unseeded_events, unseeded = scripted_decode([_clean()])
    unseeded_kwargs = [kwargs for kind, kwargs in unseeded_events if kind == "decode"]

    assert seeded_kwargs == unseeded_kwargs
    assert seeded_kwargs[0]["temperature"][0] == 0.0
    assert seeded_kwargs[0]["beam_size"] == transcribe_module.WHISPER_BEAM_SIZE
    assert unseeded["transcription"] == seeded["transcription"]
    assert unseeded["timings"] == seeded["timings"]


def test_two_runs_of_the_same_guid_agree(scripted_decode):
    """The reproducibility claim, at the level these tests can see it: same GUID, same
    seed, same published text and timings. The GPU-level confirmation that the primary
    is bit-identical belongs to the corpus run, not here."""
    first_events, first = scripted_decode([_anomalous(), _clean()])
    second_events, second = scripted_decode([_anomalous(), _clean()])
    assert first["seed"] == second["seed"]
    assert first["transcription"] == second["transcription"]
    assert first["timings"] == second["timings"]
    assert [e for e in first_events if e[0] == "decode"] == [
        e for e in second_events if e[0] == "decode"
    ]
