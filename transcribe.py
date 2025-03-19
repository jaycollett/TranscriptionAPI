import os
import logging
import warnings
import torch
import sqlite3
import math
import numpy as np
import json
from faster_whisper import WhisperModel
from pydub import AudioSegment
import signal
from functools import partial
from config import config
import time
import concurrent.futures
from typing import Dict, List, Optional, Any
import threading
import subprocess

# Configure logging with concise format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress Whisper's FP16 warning
warnings.filterwarnings("ignore", category=UserWarning, module="faster_whisper")

# Initialize device and compute type
device = config.WHISPER_DEVICE if hasattr(config, "WHISPER_DEVICE") else "cuda" if torch.cuda.is_available() else "cpu"
compute_type = config.WHISPER_COMPUTE_TYPE if hasattr(config, "WHISPER_COMPUTE_TYPE") else "float32" if device == "cuda" else "int8"  # Use int8 for CPU

# Use values from config
MAX_TRANSCRIPTION_TIME = config.MAX_TRANSCRIPTION_TIME
MAX_AUDIO_DURATION = getattr(config, "MAX_AUDIO_DURATION", 7200)  # Default: 2 hours maximum audio duration

def preprocess_audio(file_path: str) -> str:
    """Preprocess audio for optimal transcription quality."""
    try:
        # Load audio
        audio = AudioSegment.from_file(file_path)
        
        # Normalize audio levels
        audio = audio.normalize()
        
        # Remove DC offset
        audio = audio - audio.dBFS
        
        # Apply noise reduction if needed
        if audio.dBFS < -30:  # If audio is very quiet
            audio = audio + 10  # Boost by 10dB
        
        # Convert to WAV for consistent processing
        wav_path = file_path.rsplit('.', 1)[0] + '_processed.wav'
        audio.export(wav_path, format='wav')
        
        return wav_path
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {str(e)}")
        return file_path  # Return original file if preprocessing fails

def load_whisper_model():
    """Load the Whisper model with specified settings."""
    try:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but was requested. GPU support is required.")
            
        logger.info(f"Loading Whisper model on {device}: {config.WHISPER_MODEL} with {compute_type} compute type")
        
        model = WhisperModel(
            model_size_or_path=config.WHISPER_MODEL,
            device=device,
            compute_type=compute_type
        )
        logger.info(f"Whisper model loaded successfully on {device}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load Whisper model on {device}: {str(e)}")
        if device == "cuda":
            logger.error("GPU support is required. Please ensure CUDA and cuDNN are properly installed.")
            raise RuntimeError(f"Failed to initialize GPU model: {str(e)}")
        else:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

def calculate_confidence_score(segments: List[Dict[str, Any]]) -> float:
    """Calculate a weighted confidence score for the transcription."""
    if not segments:
        return 0.0
    
    # Calculate segment-level confidence based on available metrics
    segment_confidences = []
    segment_durations = []
    segment_logprobs = []
    total_duration = 0.0
    
    for segment in segments:
        duration = segment["end"] - segment["start"]
        if duration <= 0:
            continue  # Skip invalid segments
            
        # Collect log probabilities if available for more accurate calculation
        if "avg_logprob" in segment:
            segment_logprobs.append((segment["avg_logprob"], duration))
            
        # Get confidence from various sources with priority
        if "confidence" in segment:
            confidence = segment["confidence"]
        elif "avg_logprob" in segment:
            # Convert log probability to confidence
            # Whisper log probs typically range from -1 (good) to -2 (ok) to < -3 (poor)
            # Higher values (closer to 0) represent higher confidence
            avg_logprob = segment["avg_logprob"]
            
            # Map log probability range to confidence scores with more stringent requirements
            # -0.4 or better -> 0.97-1.0 confidence (excellent)
            # -0.8 -> 0.9-0.97 confidence (good)
            # -1.5 -> 0.8-0.9 confidence (acceptable)
            # -2.5 -> 0.6-0.8 confidence (marginal)
            # -3.0 or worse -> 0.4-0.6 confidence (poor)
            if avg_logprob >= -0.4:
                confidence = 0.97 + (avg_logprob / 15)  # Range: 0.97-1.0
            elif avg_logprob >= -0.8:
                confidence = 0.9 + ((avg_logprob + 0.8) * 0.175)  # Range: 0.9-0.97
            elif avg_logprob >= -1.5:
                confidence = 0.8 + ((avg_logprob + 1.5) * 0.14)  # Range: 0.8-0.9
            elif avg_logprob >= -2.5:
                confidence = 0.6 + ((avg_logprob + 2.5) * 0.2)  # Range: 0.6-0.8
            elif avg_logprob >= -3.0:
                confidence = 0.4 + ((avg_logprob + 3.0) * 0.4)  # Range: 0.4-0.6
            else:
                confidence = max(0.2, 0.4 + ((avg_logprob + 3.0) * 0.1))  # Minimum 0.2
        else:
            # Default fallback - conservative estimate
            confidence = 0.5
        
        # Adjust for no_speech_probability if available
        if "no_speech_prob" in segment:
            # Reduce confidence if high probability of no speech
            no_speech_factor = 1.0 - min(segment["no_speech_prob"], 0.9)  # Cap reduction at 90%
            confidence *= no_speech_factor
        
        # Check for potential hallucinations - very short segments with high confidence are suspicious
        if duration < 0.3 and confidence > 0.8 and len(segment.get("text", "").split()) > 2:
            confidence *= 0.8  # Penalize potential hallucinations
        
        segment_confidences.append(confidence)
        segment_durations.append(duration)
        total_duration += duration
    
    if not segment_confidences or total_duration <= 0:
        return 0.0
        
    # Calculate duration-weighted average confidence
    weighted_confidence = sum(conf * dur for conf, dur in zip(segment_confidences, segment_durations)) / total_duration
    
    # Additional scoring factors
    
    # 1. Consistency in log probability (lower variance is better)
    if len(segment_logprobs) >= 2:
        # Calculate variance in log probabilities (weighted by duration)
        total_logprob_duration = sum(dur for _, dur in segment_logprobs)
        if total_logprob_duration > 0:
            weighted_mean_logprob = sum(logp * dur for logp, dur in segment_logprobs) / total_logprob_duration
            weighted_variance = sum(dur * (logp - weighted_mean_logprob)**2 for logp, dur in segment_logprobs) / total_logprob_duration
            
            # Apply consistency bonus/penalty (up to ±5%)
            # Low variance (consistent quality) gets a bonus
            # High variance (inconsistent quality) gets a penalty
            consistency_factor = 1.0 - (min(weighted_variance, 0.5) * 0.1)
            weighted_confidence *= consistency_factor
    
    # 2. Coverage check - ensure transcription covers most of the audio
    if segments:
        coverage_ratio = total_duration / max(1.0, sum(1 for _ in segments if _.get("text", "").strip()))
        if coverage_ratio < 0.8:  # Less than 80% coverage
            # Apply coverage penalty
            coverage_factor = 0.9 + (coverage_ratio * 0.1)  # Range: 0.9-1.0
            weighted_confidence *= coverage_factor
    
    # Return confidence as percentage between 0-100
    return max(0.0, min(100.0, weighted_confidence * 100))

def transcribe_with_timeout(audio_path: str, params: Dict[str, Any], timeout: Optional[int], model: WhisperModel) -> Dict[str, Any]:
    """
    Run transcription with timeout using threading.Timer instead of signal handlers.
    """
    if timeout is None:
        timeout = config.TRANSCRIPTION_TIMEOUT
        
    if model is None:
        raise RuntimeError("Whisper model not initialized. Please ensure the model is loaded at startup.")
        
    result = {"success": False, "error": None, "transcription": None, "timings": None}
    stop_event = threading.Event()
    
    def timeout_handler():
        stop_event.set()
        logger.warning(f"Transcription timed out after {timeout} seconds")
    
    def transcription_task():
        try:
            if stop_event.is_set():
                return
                
            # Merge default parameters with user-provided parameters
            # Only include parameters that are supported by the Whisper model
            default_params = {
                "beam_size": 10,
                "best_of": 5,
                "temperature": 0.0,
                "word_timestamps": True,
                "condition_on_previous_text": True,  # Always true for better accuracy
                "compression_ratio_threshold": 1.2,
                "no_speech_threshold": 0.6,
                "language": None  # Let model auto-detect language
            }
            merged_params = {**default_params, **params}
            
            if stop_event.is_set():
                return
                
            # Try transcription with auto-detected language
            # faster-whisper returns a tuple of (segments, info)
            segments, info = model.transcribe(audio_path, **merged_params)
            
            if stop_event.is_set():
                return
                
            # Convert segments to list and extract text
            segments_list = list(segments)
            text = " ".join(segment.text for segment in segments_list)
            
            # Convert segments to dictionary format
            segments_dict = []
            for segment in segments_list:
                # Get all available attributes from segment object
                segment_attrs = [attr for attr in dir(segment) if not attr.startswith('_') and not callable(getattr(segment, attr))]
                logger.debug(f"Segment attributes: {segment_attrs}")
                
                # Extract confidence metrics from segment
                # faster-whisper Segment objects typically have these attributes:
                # - avg_logprob: Log probability per token
                # - no_speech_prob: Probability that the segment doesn't contain speech
                # - temperature: Decoding temperature used
                # - compression_ratio: Text compression ratio
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
                
                # Add all available metrics to the segment dictionary
                for attr in segment_attrs:
                    if attr not in ["start", "end", "text"] and hasattr(segment, attr):
                        value = getattr(segment, attr)
                        # Only add numeric or string values, skip complex objects
                        if isinstance(value, (int, float, str, bool)):
                            segment_dict[attr] = value
                            
                # Ensure we have word-level timestamps if available
                if hasattr(segment, "words") and segment.words:
                    segment_dict["words"] = [
                        {"start": word.start, "end": word.end, "word": word.word, 
                         "probability": word.probability if hasattr(word, "probability") else None}
                        for word in segment.words
                    ]
                    
                # Calculate confidence from log probability using improved mapping
                if "avg_logprob" in segment_dict:
                    # Whisper log probs typically range from -1 (good) to -2 (ok) to < -3 (poor)
                    avg_logprob = segment_dict["avg_logprob"]
                    
                    # Map log probability range to confidence scores with more stringent requirements
                    # -0.4 or better -> 0.97-1.0 confidence (excellent)
                    # -0.8 -> 0.9-0.97 confidence (good)
                    # -1.5 -> 0.8-0.9 confidence (acceptable)
                    # -2.5 -> 0.6-0.8 confidence (marginal)
                    # -3.0 or worse -> 0.4-0.6 confidence (poor)
                    if avg_logprob >= -0.4:
                        confidence = 0.97 + (avg_logprob / 15)  # Range: 0.97-1.0
                    elif avg_logprob >= -0.8:
                        confidence = 0.9 + ((avg_logprob + 0.8) * 0.175)  # Range: 0.9-0.97
                    elif avg_logprob >= -1.5:
                        confidence = 0.8 + ((avg_logprob + 1.5) * 0.14)  # Range: 0.8-0.9
                    elif avg_logprob >= -2.5:
                        confidence = 0.6 + ((avg_logprob + 2.5) * 0.2)  # Range: 0.6-0.8
                    elif avg_logprob >= -3.0:
                        confidence = 0.4 + ((avg_logprob + 3.0) * 0.4)  # Range: 0.4-0.6
                    else:
                        confidence = max(0.2, 0.4 + ((avg_logprob + 3.0) * 0.1))  # Minimum 0.2
                        
                    segment_dict["confidence"] = confidence
                elif hasattr(segment, "confidence"):
                    segment_dict["confidence"] = segment.confidence
                else:
                    segment_dict["confidence"] = 0.5  # Default fallback
                    
                # Apply additional confidence adjustments based on segment properties
                # Adjust for no_speech_probability
                if "no_speech_prob" in segment_dict and segment_dict["no_speech_prob"] > 0.3:
                    no_speech_factor = 1.0 - min(segment_dict["no_speech_prob"], 0.9)  # Cap reduction at 90%
                    segment_dict["confidence"] *= no_speech_factor
                    
                # Adjust for potentially hallucinated short segments
                segment_duration = segment_dict["end"] - segment_dict["start"]
                words_count = len(segment_dict["text"].split())
                if segment_duration < 0.3 and words_count > 1 and segment_dict["confidence"] > 0.8:
                    segment_dict["confidence"] *= 0.8  # Reduce confidence for suspiciously short segments
                
                segments_dict.append(segment_dict)
            
            
            result["success"] = True
            result["transcription"] = text
            result["timings"] = segments_dict
            
        except Exception as e:
            if not stop_event.is_set():
                result["error"] = str(e)
                logger.error(f"Transcription error: {e}")
    
    # Start the transcription in a separate thread
    transcription_thread = threading.Thread(target=transcription_task)
    transcription_thread.start()
    
    # Set up the timeout timer
    timer = threading.Timer(timeout, timeout_handler)
    timer.start()
    
    # Wait for either completion or timeout
    transcription_thread.join(timeout=timeout)
    
    # Clean up
    timer.cancel()
    
    if stop_event.is_set():
        result["error"] = "Transcription timed out"
        return result
        
    if not result["success"]:
        result["error"] = "Transcription failed"
        
    return result

def transcribe_audio(audio_path: str, guid: str, model) -> Dict[str, Any]:
    """
    Transcribe audio with multiple passes and confidence scoring.
    Optimized for sermon-length audio (10-45 minutes).
    """
    if model is None:
        raise RuntimeError("Whisper model not initialized. Please ensure the model is loaded at startup.")
    
    logger.info(f"Starting transcription for: {audio_path}")
    
    # Load audio file using the imported AudioSegment
    audio = AudioSegment.from_file(audio_path)
    duration_sec = len(audio) / 1000
    logger.info(f"Audio duration: {duration_sec:.2f} seconds")
    
    # Calculate timeout based on audio duration
    # For sermons (10-45 minutes), we need more time for forced alignment
    # Base transcription time: 1/10th of audio duration
    # Forced alignment time: 1/5th of audio duration
    # Add buffer for multiple passes and processing overhead
    base_timeout = int(duration_sec / 10)  # Base transcription time
    alignment_timeout = int(duration_sec / 5)  # Forced alignment time
    processing_overhead = getattr(config, "PROCESSING_OVERHEAD", 360)  # Default: 6 minutes for multiple passes and processing
    
    # Total timeout calculation
    min_timeout = getattr(config, "MIN_TRANSCRIPTION_TIMEOUT", 600)  # Default: at least 10 minutes
    timeout = min(
        max(min_timeout, base_timeout + alignment_timeout + processing_overhead),
        config.MAX_TRANSCRIPTION_TIME
    )
    
    logger.info(f"Calculated timeout: {timeout} seconds (transcription: {base_timeout}s, alignment: {alignment_timeout}s, overhead: {processing_overhead}s)")
    
    # Define transcription passes optimized for sermon-length audio
    passes = {
        "highest_accuracy": {
            "beam_size": 10,     # Maximum accuracy with large beam
            "best_of": 10,       # Match beam_size
            "temperature": 0.0,  # Fully deterministic for maximum consistency
            "word_timestamps": True,
            "condition_on_previous_text": True,
            "compression_ratio_threshold": 1.2,  # Standard threshold
            "no_speech_threshold": 0.5,  # More sensitive to speech detection
            "initial_prompt": None  # No initial prompt for first pass
        },
        "high_accuracy": {
            "beam_size": 7,     # High accuracy with moderate beam
            "best_of": 7,       # Match beam_size
            "temperature": 0.1,  
            "word_timestamps": True,
            "condition_on_previous_text": True,
            "compression_ratio_threshold": 1.1,
            "no_speech_threshold": 0.55
        },
        "balanced": {
            "beam_size": 7,     # Same beam size but with temperature variation
            "best_of": 7,       # Match beam_size
            "temperature": 0.0,  
            "word_timestamps": True,
            "condition_on_previous_text": True,
            "compression_ratio_threshold": 1.1,
            "no_speech_threshold": 0.6
        },
        "moderate": {
            "beam_size": 5,      # Moderate beam
            "best_of": 5,        # Match beam_size
            "temperature": 0.0,  # Fully deterministic
            "word_timestamps": True,
            "condition_on_previous_text": True,
            "compression_ratio_threshold": 1.0,  # Stricter threshold
            "no_speech_threshold": 0.55
        }
    }
    
    best_result = None
    best_confidence = 0.0
    pass_results = []
    
    # Try each pass and collect results
    for pass_name, params in passes.items():
        logger.info(f"Starting {pass_name} pass...")
        
        result = transcribe_with_timeout(audio_path, params, timeout, model)
        
        if result["success"]:
            # Calculate confidence score
            confidence = calculate_confidence_score(result["timings"])
            num_segments = len(result["timings"])
            total_duration = sum(seg["end"] - seg["start"] for seg in result["timings"])
            avg_segment_length = total_duration / num_segments if num_segments > 0 else 0
            
            logger.info(f"{pass_name} pass completed successfully with confidence score of: {confidence:.3f}%")
            
            # Store result with metadata
            pass_results.append({
                "pass_name": pass_name,
                "result": result,
                "confidence": confidence,
                "duration": duration_sec
            })
        else:
            logger.error(f"{pass_name} pass failed: {result.get('error', 'Unknown error')}")
    
    if not pass_results:
        raise RuntimeError("All transcription passes failed")
    
    # Sort results by confidence
    pass_results.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Select the highest confidence result
    best_result = pass_results[0]["result"]
    best_confidence = pass_results[0]["confidence"]
    logger.info(f"Selected {pass_results[0]['pass_name']} pass as best result:")
    logger.info(f"  - Confidence score: {best_confidence:.3f}%")
    logger.info(f"  - Number of segments: {len(best_result['timings'])}")
    logger.info(f"  - Total duration: {sum(seg['end'] - seg['start'] for seg in best_result['timings']):.2f}s")
    
    # If the highest confidence is below our threshold, try additional techniques
    confidence_threshold = getattr(config, "CONFIDENCE_THRESHOLD", 95.0)
    if best_confidence < confidence_threshold:
        logger.info(f"Highest confidence below threshold ({confidence_threshold}%), trying enhancement techniques")
        
        # 1. First try an adaptive pass with initial prompt from best result
        best_text = best_result["transcription"]
        initial_prompt = best_text[:500] if len(best_text) > 500 else best_text
        
        adaptive_params = {
            "beam_size": 12,    # High beam size for accuracy but not excessive
            "best_of": 12,      # Match beam_size
            "temperature": 0.0,  # Stay deterministic for consistency
            "word_timestamps": True,
            "condition_on_previous_text": True,
            "initial_prompt": initial_prompt,  # Use best result text as prompt
            "compression_ratio_threshold": getattr(config, "COMPRESSION_RATIO_THRESHOLD", 1.15),  # Slightly tighter
            "no_speech_threshold": getattr(config, "NO_SPEECH_THRESHOLD", 0.5)  # More sensitive to speech
        }
        
        logger.info("Adaptive pass with initial prompt settings:")
        logger.info(f"  - Beam size: {adaptive_params['beam_size']}")
        logger.info(f"  - Initial prompt: Using {len(adaptive_params['initial_prompt'])} characters from best result")
        
        adaptive_result = transcribe_with_timeout(audio_path, adaptive_params, timeout, model)
        
        # 2. Try a VAD-based preprocessing pass to reduce noise/silence in difficult sections
        # Prepare a VAD-enhanced version of audio for challenging sections
        vad_enhanced_audio_path = None
        try:
            # Use the already imported AudioSegment
            from pydub import silence
            from tempfile import NamedTemporaryFile
            
            # Only process if confidence is still below threshold or audio has significant non-speech
            if (adaptive_result["success"] and 
                calculate_confidence_score(adaptive_result["timings"]) < confidence_threshold):
                
                logger.info("Preparing VAD-enhanced audio for challenging sections")
                
                # Load audio file
                audio = AudioSegment.from_file(audio_path)
                
                # Detect non-speech segments
                non_speech_segments = []
                for segment in best_result["timings"]:
                    if "no_speech_prob" in segment and segment["no_speech_prob"] > 0.7:
                        start_ms = int(segment["start"] * 1000)
                        end_ms = int(segment["end"] * 1000)
                        non_speech_segments.append((start_ms, end_ms))
                
                # Apply VAD to refine non-speech detection
                # Minimum silence length of 500ms, silence threshold of -35dBFS
                silence_ranges = silence.detect_silence(audio, min_silence_len=500, silence_thresh=-35)
                
                # Combine detected silence ranges with high no_speech_prob segments
                all_silence = list(silence_ranges)
                for start_ms, end_ms in non_speech_segments:
                    all_silence.append((start_ms, end_ms))
                
                # Sort by start time and merge overlapping ranges
                if all_silence:
                    all_silence.sort(key=lambda x: x[0])
                    merged_silence = [all_silence[0]]
                    
                    for start_ms, end_ms in all_silence[1:]:
                        last_start, last_end = merged_silence[-1]
                        if start_ms <= last_end + 200:  # Merge if less than 200ms gap
                            merged_silence[-1] = (last_start, max(last_end, end_ms))
                        else:
                            merged_silence.append((start_ms, end_ms))
                    
                    # Create enhanced audio by attenuating silence instead of removing
                    # This maintains audio length while reducing noise
                    enhanced_audio = audio.copy()
                    for start_ms, end_ms in merged_silence:
                        # Extract silence
                        silence_segment = audio[start_ms:end_ms]
                        # Attenuate by 80% (-14dB)
                        attenuated = silence_segment - 14
                        # Replace in enhanced audio
                        enhanced_audio = enhanced_audio[:start_ms] + attenuated + enhanced_audio[end_ms:]
                    
                    # Write to temporary file
                    with NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        enhanced_audio.export(f.name, format="wav")
                        vad_enhanced_audio_path = f.name
                        logger.info(f"VAD-enhanced audio created at {vad_enhanced_audio_path}")
        except Exception as e:
            logger.warning(f"VAD enhancement failed: {str(e)}")
        # 3. Try VAD-enhanced audio with adaptive parameters
        vad_result = None
        if vad_enhanced_audio_path:
            vad_params = adaptive_params.copy()
            logger.info("Running transcription on VAD-enhanced audio")
            vad_result = transcribe_with_timeout(vad_enhanced_audio_path, vad_params, timeout, model)
        
        # 4. Try a chunking approach for long audio
        chunked_result = None
        if duration_sec > 300:  # For audio longer than 5 minutes
            try:
                logger.info("Attempting chunked transcription for long audio")
                # Create a temporary transcript with concatenated segments
                # from previous passes to guide chunking
                concatenated_texts = []
                if best_result["success"]:
                    concatenated_texts.append(best_result["transcription"])
                if adaptive_result["success"]:
                    concatenated_texts.append(adaptive_result["transcription"])
                
                chunked_segments = []
                
                # Roughly determine chunk boundaries based on silence or punctuation
                chunk_size = 120  # Aim for ~2 minute chunks
                chunk_points = []
                
                # Get best timestamps to use as chunk points
                timestamps = best_result["timings"]
                if len(timestamps) > 5:
                    # Find natural break points (silences, periods, etc.)
                    for i, seg in enumerate(timestamps[1:-1], 1):
                        text = seg["text"]
                        # Good breaking points are at sentence boundaries
                        if (text.strip().endswith(('.', '?', '!')) or 
                            ("no_speech_prob" in seg and seg["no_speech_prob"] > 0.5)):
                            chunk_points.append((i, seg["end"]))
                    
                    # Select chunk points to aim for ~120 second chunks
                    if chunk_points:
                        sorted_points = sorted(chunk_points, key=lambda x: x[1])
                        selected_points = []
                        last_point = 0
                        
                        for idx, time_point in sorted_points:
                            if time_point - last_point >= chunk_size:
                                selected_points.append(idx)
                                last_point = time_point
                        
                        # Process each chunk
                        if selected_points:
                            chunks = []
                            start_idx = 0
                            
                            for end_idx in selected_points:
                                chunks.append((start_idx, end_idx))
                                start_idx = end_idx
                            
                            # Add final chunk
                            chunks.append((start_idx, len(timestamps)))
                            
                            # Process each chunk
                            for chunk_start, chunk_end in chunks:
                                chunk_segments = timestamps[chunk_start:chunk_end]
                                start_time = chunk_segments[0]["start"]
                                end_time = chunk_segments[-1]["end"]
                                chunk_duration = end_time - start_time
                                
                                if chunk_duration < 5:  # Skip very short chunks
                                    chunked_segments.extend(chunk_segments)
                                    continue
                                
                                # Get context from surrounding text
                                context = ""
                                if chunk_start > 0:
                                    # Use previous chunk as context
                                    prev_segments = timestamps[max(0, chunk_start-5):chunk_start]
                                    context = " ".join(s["text"] for s in prev_segments)
                                    context = context[-500:] if len(context) > 500 else context
                                
                                # Create chunk parameters
                                chunk_params = {
                                    "beam_size": 15,  # Balance of speed and accuracy
                                    "best_of": 15,    # Match beam_size
                                    "temperature": 0.0,
                                    "word_timestamps": True,
                                    "condition_on_previous_text": True,
                                    "initial_prompt": context,
                                    "compression_ratio_threshold": 1.15,  # More balanced
                                    "no_speech_threshold": 0.5  # More sensitive to catch all speech
                                }
                                
                                # Need to extract the audio chunk
                                # Use the already imported AudioSegment
                                with NamedTemporaryFile(suffix=".wav", delete=False) as f:
                                    audio = AudioSegment.from_file(audio_path)
                                    start_ms = int(start_time * 1000)
                                    end_ms = int(end_time * 1000)
                                    chunk_audio = audio[start_ms:end_ms]
                                    chunk_audio.export(f.name, format="wav")
                                    
                                    # Process this chunk
                                    logger.info(f"Processing chunk {chunk_start}-{chunk_end} ({start_time:.2f}s-{end_time:.2f}s)")
                                    chunk_result = transcribe_with_timeout(f.name, chunk_params, timeout//2, model)
                                    
                                    if chunk_result["success"]:
                                        # Adjust timestamps to match original audio
                                        for segment in chunk_result["timings"]:
                                            segment["start"] += start_time
                                            segment["end"] += start_time
                                        chunked_segments.extend(chunk_result["timings"])
                                    else:
                                        # Fallback to original segments if chunk fails
                                        chunked_segments.extend(chunk_segments)
                                        
                            if chunked_segments:
                                # Sort by start time in case of overlap
                                chunked_segments.sort(key=lambda x: x["start"])
                                chunked_result = {
                                    "success": True,
                                    "timings": chunked_segments,
                                    "transcription": " ".join(s["text"] for s in chunked_segments),
                                }
            except Exception as e:
                logger.warning(f"Chunking approach failed: {str(e)}")
                        
        # Evaluate all results and select the best
        all_results = []
        
        if best_result["success"]:
            best_confidence = calculate_confidence_score(best_result["timings"]) 
            all_results.append(("original", best_result, best_confidence))
            
        if adaptive_result and adaptive_result["success"]:
            adaptive_confidence = calculate_confidence_score(adaptive_result["timings"])
            all_results.append(("adaptive", adaptive_result, adaptive_confidence))
            
        if vad_result and vad_result["success"]:
            vad_confidence = calculate_confidence_score(vad_result["timings"])
            all_results.append(("vad", vad_result, vad_confidence))
            
        if chunked_result:
            chunked_confidence = calculate_confidence_score(chunked_result["timings"])
            all_results.append(("chunked", chunked_result, chunked_confidence))
            
        # Sort by confidence and select the best
        all_results.sort(key=lambda x: x[2], reverse=True)
        for method, result, confidence in all_results:
            logger.info(f"{method.title()} method confidence: {confidence:.2f}%")
            
        if all_results:
            best_method, best_result, best_confidence = all_results[0]
            logger.info(f"Selected {best_method} method with confidence: {best_confidence:.2f}%")
            
        # Clean up temporary files
        if vad_enhanced_audio_path:
            try:
                os.remove(vad_enhanced_audio_path)
            except:
                pass
    else:
        logger.info(f"Confidence already above threshold ({best_confidence:.2f}% >= {confidence_threshold}%), skipping enhancement")
    
    # Run forced alignment on the best result
    logger.info("Running forced alignment to refine timestamps")
    try:
        refined_segments = run_forced_alignment(audio_path, best_result["timings"], guid)
        
        # Preserve confidence scores when updating with forced alignment
        # Copy confidence scores from original segments to aligned segments if they're missing
        for i, refined_segment in enumerate(refined_segments):
            if i < len(best_result["timings"]) and "confidence" not in refined_segment and "confidence" in best_result["timings"][i]:
                refined_segment["confidence"] = best_result["timings"][i]["confidence"]
            # Also preserve other metrics if available
            for metric in ["avg_logprob", "no_speech_prob"]:
                if i < len(best_result["timings"]) and metric not in refined_segment and metric in best_result["timings"][i]:
                    refined_segment[metric] = best_result["timings"][i][metric]
        
        best_result["timings"] = refined_segments
        logger.info("Forced alignment completed successfully")
    except Exception as e:
        logger.error(f"Forced alignment failed: {str(e)}")
        # Continue with original timings if forced alignment fails
    
    # Add the confidence score to the result
    best_result["confidence"] = best_confidence
    logger.info(f"Final confidence score: {best_confidence:.2f}%")
    
    return best_result

def run_forced_alignment(audio_path: str, whisper_segments: List[Dict[str, Any]], guid: str) -> List[Dict[str, Any]]:
    """
    Runs Montreal Forced Aligner (MFA) to refine the timestamps from the Whisper transcript,
    ensuring they align with Whisper's segment structure.
    """
    upload_folder = config.UPLOAD_FOLDER
    transcript_path = os.path.join(upload_folder, f"{guid}.txt")
    aligned_output_dir = os.path.join(upload_folder, f"{guid}_aligned")
    alignment_json_path = os.path.join(aligned_output_dir, f"{guid}.json")

    # If alignment already exists, avoid re-running MFA
    if os.path.exists(alignment_json_path):
        logger.info(f"MFA alignment exists for {guid}, skipping")
    else:
        try:
            # Save transcript to a file for MFA (remove extra spaces)
            with open(transcript_path, "w") as f:
                f.write(" ".join(seg["text"].strip() for seg in whisper_segments if seg["text"].strip()))

            # Ensure output directory exists
            os.makedirs(aligned_output_dir, exist_ok=True)

            # Run MFA command
            mfa_command = [
                "mfa", "align",
                upload_folder,
                "/mfa/pretrained_models/dictionary/english_mfa.dict",
                "english_mfa",
                aligned_output_dir,
                "--output_format", "json"
            ]

            result = subprocess.run(mfa_command, check=True, capture_output=True, text=True)

            if not os.path.exists(alignment_json_path):
                logger.error(f"MFA output file not found: {alignment_json_path}")
                return whisper_segments

        except subprocess.CalledProcessError as e:
            logger.error(f"MFA alignment failed: {e.stderr}")
            return whisper_segments

        except Exception as e:
            logger.error(f"Unexpected error running MFA: {str(e)}")
            return whisper_segments

    # Read MFA output file
    try:
        with open(alignment_json_path, "r") as f:
            alignment_data = json.load(f)

        # Ensure MFA output format is valid
        if "tiers" not in alignment_data or "words" not in alignment_data["tiers"]:
            logger.error(f"Words tier missing in MFA output")
            return whisper_segments

        # Extract word-level alignments
        word_entries = alignment_data["tiers"]["words"]["entries"]
        words = [{"start": entry[0], "end": entry[1], "text": entry[2]} for entry in word_entries]

        # Align MFA words within Whisper's segment-level structure
        refined_segments = []
        for segment in whisper_segments:
            whisper_start, whisper_end, segment_text = segment["start"], segment["end"], segment["text"]

            # Find words within the Whisper segment boundary
            segment_words = [word for word in words if whisper_start <= word["start"] <= whisper_end]

            if segment_words:
                refined_start = segment_words[0]["start"]
                refined_end = segment_words[-1]["end"]
            else:
                refined_start, refined_end = whisper_start, whisper_end

            refined_segments.append({
                "start": refined_start,
                "end": refined_end,
                "text": segment_text
            })

        return refined_segments

    except Exception as e:
        logger.error(f"Error processing MFA output: {str(e)}")
        return whisper_segments

