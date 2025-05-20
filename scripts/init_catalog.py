#!/usr/bin/env python
"""
Initialize the asset catalog database for tracking data files.
"""
import os
import sqlite3
import hashlib
import datetime
from pathlib import Path

def init_catalog(db_path="catalog.db"):
    """Initialize the catalog database if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create assets table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS assets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL UNIQUE,
        hash TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        created_at TIMESTAMP NOT NULL,
        updated_at TIMESTAMP NOT NULL,
        metadata TEXT
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Catalog initialized at {db_path}")

def compute_file_hash(file_path, algorithm='sha256'):
    """Compute hash for a file."""
    file_path = Path(file_path)
    hash_obj = hashlib.new(algorithm)
    
    # Handle directories (like zarr stores)
    if file_path.is_dir():
        for subpath in sorted(file_path.rglob('*')):
            if subpath.is_file():
                hash_obj.update(subpath.name.encode())
                with open(subpath, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b''):
                        hash_obj.update(chunk)
    else:
        # Regular file handling
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_obj.update(chunk)
                
    return hash_obj.hexdigest()

def register_asset(file_path, metadata=None, db_path="catalog.db"):
    """
    Register a file or directory in the catalog database.
    
    Args:
        file_path: Path to the file or directory
        metadata: Optional JSON-serializable metadata
        db_path: Path to the catalog database
    
    Returns:
        True if registration was successful, False if file already exists with same hash
    """
    import json
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_hash = compute_file_hash(file_path)
    file_size = sum(f.stat().st_size for f in file_path.rglob('*') if f.is_file()) if file_path.is_dir() else file_path.stat().st_size
    timestamp = datetime.datetime.now().isoformat()
    
    # Convert metadata to JSON string if it's a dict
    if metadata is not None:
        if isinstance(metadata, dict):
            metadata = json.dumps(metadata)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if file with same path exists
    cursor.execute("SELECT hash FROM assets WHERE path = ?", (str(file_path),))
    existing = cursor.fetchone()
    
    if existing:
        existing_hash = existing[0]
        if existing_hash == file_hash:
            print(f"File already registered with same hash: {file_path}")
            conn.close()
            return False
        
        # Update the existing record
        cursor.execute('''
        UPDATE assets 
        SET hash = ?, size_bytes = ?, updated_at = ?, metadata = ?
        WHERE path = ?
        ''', (file_hash, file_size, timestamp, metadata, str(file_path)))
    else:
        # Insert new record
        cursor.execute('''
        INSERT INTO assets (path, hash, size_bytes, created_at, updated_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (str(file_path), file_hash, file_size, timestamp, timestamp, metadata))
    
    conn.commit()
    conn.close()
    print(f"Registered asset: {file_path}")
    return True

if __name__ == "__main__":
    init_catalog()
    print("Catalog database initialized. Use register_asset() to add files.")