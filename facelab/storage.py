"""Local SQLite storage. No original photos or arbitrary file paths."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS people (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, name_key TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL, thumbnail BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY, person_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    image_hash TEXT UNIQUE NOT NULL, embedding TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY, image_hash TEXT UNIQUE NOT NULL, payload TEXT NOT NULL);
            ''')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def people(self):
        with self.connection() as db:
            return [dict(r) for r in db.execute('''
                SELECT p.id,p.name,p.created_at,COUNT(s.id) AS samples
                FROM people p LEFT JOIN samples s ON p.id=s.person_id
                GROUP BY p.id ORDER BY p.created_at,p.id''')]

    def name_exists(self, name_key):
        with self.connection() as db:
            return db.execute('SELECT 1 FROM people WHERE name_key=?', (name_key,)).fetchone() is not None

    def reference_hashes(self):
        with self.connection() as db:
            return {r[0] for r in db.execute('SELECT image_hash FROM samples')}

    def gallery(self):
        gallery = {}
        with self.connection() as db:
            for row in db.execute('SELECT person_id,embedding FROM samples ORDER BY id'):
                gallery.setdefault(row['person_id'], []).append(json.loads(row['embedding']))
        return gallery

    def add(self, name, name_key, thumbnail, samples):
        person_id = uuid4().hex
        with self.connection() as db:
            db.execute('INSERT INTO people VALUES (?,?,?,?,?)',
                       (person_id,name,name_key,datetime.now(timezone.utc).isoformat(),thumbnail))
            db.executemany('INSERT INTO samples(person_id,image_hash,embedding) VALUES (?,?,?)',
                           [(person_id,h,json.dumps(v)) for h,v in samples])
            db.execute('DELETE FROM evaluations')
        return next(p for p in self.people() if p['id'] == person_id)

    def thumbnail(self, person_id):
        with self.connection() as db:
            row=db.execute('SELECT thumbnail FROM people WHERE id=?',(person_id,)).fetchone()
            return bytes(row[0]) if row else None

    def delete(self, person_id):
        with self.connection() as db:
            result = db.execute('DELETE FROM people WHERE id=?',(person_id,))
            if result.rowcount == 0:
                raise KeyError('Человек не найден.')
            db.execute('DELETE FROM evaluations')

    def evaluations(self):
        with self.connection() as db:
            return [json.loads(r[0]) for r in db.execute('SELECT payload FROM evaluations ORDER BY id')]

    def add_evaluations(self, cases):
        with self.connection() as db:
            db.executemany('INSERT INTO evaluations(image_hash,payload) VALUES (?,?)',
                           [(c['image_hash'],json.dumps(c,ensure_ascii=False,allow_nan=False)) for c in cases])

    def clear_evaluations(self):
        with self.connection() as db:
            db.execute('DELETE FROM evaluations')
