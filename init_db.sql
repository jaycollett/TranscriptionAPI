CREATE TABLE IF NOT EXISTS transcriptions (
    guid TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    transcription TEXT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
