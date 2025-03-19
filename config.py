import os
from typing import Dict, Any, Set
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "5000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # File Storage
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "/data/audio_files")
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "524288000"))  # 500MB
    ALLOWED_AUDIO_TYPES: Set[str] = field(default_factory=lambda: {
        "audio/wav",
        "audio/mp3",
        "audio/mpeg",
        "audio/ogg",
        "audio/x-wav"
    })
    
    # Database Configuration
    DB_FILE: str = os.getenv("DB_FILE", "/data/transcriptions.db")
    MAX_DB_CONNECTIONS: int = int(os.getenv("MAX_DB_CONNECTIONS", "5"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY: int = int(os.getenv("RETRY_DELAY", "5"))
    DB_TIMEOUT: int = int(os.getenv("DB_TIMEOUT", "30"))
    
    # Processing Settings
    PROCESSING_SPEED_FACTOR: float = float(os.getenv("PROCESSING_SPEED_FACTOR", "15.1"))  # For fast processing
    MAX_TRANSCRIPTION_TIME: int = int(os.getenv("MAX_TRANSCRIPTION_TIME", "3600"))  # 1 hour
    TRANSCRIPTION_TIMEOUT: int = int(os.getenv("TRANSCRIPTION_TIMEOUT", "300"))  # 5 minutes per pass
    
    # Cleanup Settings
    CLEANUP_AGE_DAYS: int = int(os.getenv("CLEANUP_AGE_DAYS", "1"))
    CLEANUP_BATCH_SIZE: int = int(os.getenv("CLEANUP_BATCH_SIZE", "100"))
    CLEANUP_INTERVAL: int = int(os.getenv("CLEANUP_INTERVAL", "3600"))  # 1 hour
    
    # Whisper Model Settings
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3-turbo")  # Using turbo variant
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cuda")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "float32")  # Using float32 for stability
    
    # Transcription Parameters
    VAD_FILTER: bool = os.getenv("VAD_FILTER", "true").lower() == "true"
    MIN_SILENCE_DURATION_MS: int = int(os.getenv("MIN_SILENCE_DURATION_MS", "500"))
    CONDITION_ON_PREVIOUS_TEXT: bool = os.getenv("CONDITION_ON_PREVIOUS_TEXT", "true").lower() == "true"
    COMPRESSION_RATIO_THRESHOLD: float = float(os.getenv("COMPRESSION_RATIO_THRESHOLD", "1.2"))
    NO_SPEECH_THRESHOLD: float = float(os.getenv("NO_SPEECH_THRESHOLD", "0.6"))
    
    def __post_init__(self):
        """Validate and adjust configuration values after initialization."""
        # Ensure upload folder exists
        Path(self.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
        
        # Validate numeric values
        if self.MAX_FILE_SIZE <= 0:
            raise ValueError("MAX_FILE_SIZE must be positive")
        if self.MAX_DB_CONNECTIONS <= 0:
            raise ValueError("MAX_DB_CONNECTIONS must be positive")
        if self.MAX_RETRIES < 0:
            raise ValueError("MAX_RETRIES must be non-negative")
        if self.RETRY_DELAY < 0:
            raise ValueError("RETRY_DELAY must be non-negative")
        if self.DB_TIMEOUT <= 0:
            raise ValueError("DB_TIMEOUT must be positive")
        if self.CLEANUP_AGE_DAYS <= 0:
            raise ValueError("CLEANUP_AGE_DAYS must be positive")
        if self.CLEANUP_BATCH_SIZE <= 0:
            raise ValueError("CLEANUP_BATCH_SIZE must be positive")
        if self.CLEANUP_INTERVAL <= 0:
            raise ValueError("CLEANUP_INTERVAL must be positive")
        if self.PROCESSING_SPEED_FACTOR <= 0:
            raise ValueError("PROCESSING_SPEED_FACTOR must be positive")
        if self.MAX_TRANSCRIPTION_TIME <= 0:
            raise ValueError("MAX_TRANSCRIPTION_TIME must be positive")
        if self.TRANSCRIPTION_TIMEOUT <= 0:
            raise ValueError("TRANSCRIPTION_TIMEOUT must be positive")
        if self.MIN_SILENCE_DURATION_MS <= 0:
            raise ValueError("MIN_SILENCE_DURATION_MS must be positive")
        if self.COMPRESSION_RATIO_THRESHOLD <= 0:
            raise ValueError("COMPRESSION_RATIO_THRESHOLD must be positive")
        if not 0 <= self.NO_SPEECH_THRESHOLD <= 1:
            raise ValueError("NO_SPEECH_THRESHOLD must be between 0 and 1")
        
        # Validate timeouts are reasonable
        if self.TRANSCRIPTION_TIMEOUT > self.MAX_TRANSCRIPTION_TIME:
            raise ValueError("TRANSCRIPTION_TIMEOUT cannot be greater than MAX_TRANSCRIPTION_TIME")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary, excluding sensitive values."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and k != 'DB_FILE'
        }

# Create global configuration instance
config = Config() 