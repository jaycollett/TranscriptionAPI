"""Tests for the ETA arithmetic in the checkQueue.py helper script."""

from datetime import datetime, timedelta

import checkQueue

NOW = datetime(2026, 9, 7, 12, 0, 0)


def _job(guid, status, submitted_at, est):
    return {"guid": guid, "filename": f"{guid}.mp3", "status": status,
            "submitted_at": submitted_at, "completed_at": "", "processing_time_est": est}


def test_estimates_accumulate_in_submission_order():
    rows = [
        _job("c", "pending", "2026-09-07 11:03:00", 100),
        _job("a", "processing", "2026-09-07 11:00:00", 600),
        _job("b", "pending", "2026-09-07 11:01:00", 300),
        _job("z", "completed", "2026-09-07 10:00:00", 9999),
    ]
    estimates = checkQueue.estimate_completions(rows, now=NOW)
    assert [job["guid"] for job, _ in estimates] == ["b", "c"]
    assert estimates[0][1] == NOW + timedelta(seconds=600 + 300)
    assert estimates[1][1] == NOW + timedelta(seconds=600 + 300 + 100)


def test_no_pending_jobs_gives_empty_list():
    rows = [_job("a", "processing", "2026-09-07 11:00:00", 600)]
    assert checkQueue.estimate_completions(rows, now=NOW) == []
