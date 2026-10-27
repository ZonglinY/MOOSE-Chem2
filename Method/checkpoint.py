"""Durable API replay with stable identities for parallel search tasks."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from contextlib import contextmanager
import hashlib
import random
import os
import sys
from importlib.metadata import version
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import threading


@dataclass
class Scope:
    path: str
    calls: int = 0
    executors: int = 0


_scope = ContextVar('request_scope', default=None)
_journal = ContextVar('request_journal', default=None)


class RequestInterrupted(BaseException):
    """Stop extraction/search catch-all handlers after a failed API request."""


def checkpoint_active():
    return _journal.get() is not None


def shuffle_candidates(values):
    journal = _journal.get()
    (journal.random if journal else random).shuffle(values)


def api_response(api, request, send, response_type):
    journal = _journal.get()
    if journal is None:
        return send()
    try:
        payload = journal.call({'api': api, **request},
                               lambda: send().model_dump(mode='json'))
        return response_type.model_validate(payload)
    except Exception as exc:
        raise RequestInterrupted('Request interrupted; rerun with --resume') from exc


class ReplayExecutor(ThreadPoolExecutor):
    """Assign call scopes in submission order, independently of worker timing."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        parent = _scope.get()
        self.path = None
        self.submissions = 0
        if parent is not None:
            self.path = f'{parent.path}/pool{parent.executors}'
            parent.executors += 1

    def submit(self, fn, /, *args, **kwargs):
        if self.path is None:
            return super().submit(fn, *args, **kwargs)
        scope = Scope(f'{self.path}/task{self.submissions}')
        self.submissions += 1
        def invoke():
            token = _scope.set(scope)
            try:
                return fn(*args, **kwargs)
            finally:
                _scope.reset(token)
        context = copy_context()
        return super().submit(context.run, invoke)


class RequestJournal:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS calls ('
                        'scope TEXT, ordinal INTEGER, request TEXT NOT NULL, response TEXT, '
                        'PRIMARY KEY(scope, ordinal))')
        self.db.commit()
        self.token = _scope.set(Scope('root'))
        self.journal_token = _journal.set(self)
        self.replayed = 0
        self.sent = 0
        self.random = random.Random(0)

    def call(self, request, send):
        scope = _scope.get()
        if scope is None:
            raise RuntimeError('API call has no replay task scope')
        key = (scope.path, scope.calls)
        canonical = json.dumps(request, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        with self.lock:
            row = self.db.execute('SELECT request, response FROM calls WHERE scope=? AND ordinal=?', key).fetchone()
            if row is None:
                self.db.execute('INSERT INTO calls VALUES (?, ?, ?, NULL)', (*key, canonical))
                self.db.commit()
            elif row[0] != canonical:
                raise ValueError(f'Resume request mismatch at {key}; use the original inputs, configuration and code')
            elif row[1] is not None:
                self.replayed += 1
                scope.calls += 1
                return json.loads(row[1])
            self.sent += 1
        # A failed request keeps this slot, including retries in this process.
        response = send()
        serialized = json.dumps(response, ensure_ascii=False)
        with self.lock:
            self.db.execute('UPDATE calls SET response=? WHERE scope=? AND ordinal=?', (serialized, *key))
            self.db.commit()
        scope.calls += 1
        return response

    def close(self):
        self.db.close()
        _scope.reset(self.token)
        _journal.reset(self.journal_token)


@contextmanager
def checkpoint_run(args):
    """Own a journal and validate the original search before replaying it."""
    if not args.checkpoint:
        if args.resume:
            raise ValueError('--resume requires --checkpoint PATH')
        yield
        return
    import fcntl
    path = Path(args.checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(path) + '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Checkpoint is already in use') from exc
        if path.exists() != args.resume:
            raise ValueError('Use --resume for an existing checkpoint, or a new checkpoint path')
        config = {k: v for k, v in vars(args).items()
                  if k not in {'api_key', 'eval_api_key', 'checkpoint', 'resume'}}
        inputs = ([args.custom_research_background_and_coarse_hyp_path]
                  if args.if_use_custom_research_background_and_coarse_hyp
                  else [args.chem_annotation_path])
        if not args.if_use_custom_research_background_and_coarse_hyp and args.if_use_vague_cg_hyp_as_input:
            inputs.append(args.vague_cg_hyp_path)
        root = Path(__file__).resolve().parents[1]
        sources = sorted((root / 'Method').glob('*.py')) + [root / 'Evaluation/pairwise_compare.py']
        digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        identity = json.dumps({'config': config,
                               'inputs': {p: digest(p) for p in inputs},
                               'sources': {str(p.relative_to(root)): digest(p) for p in sources},
                               'python': sys.version,
                               'provider_environment': {name: os.environ.get(name) for name in [
                                   'GOOGLE_GENAI_USE_VERTEXAI', 'GOOGLE_GENAI_USE_ENTERPRISE',
                                   'GOOGLE_CLOUD_PROJECT', 'GOOGLE_CLOUD_LOCATION',
                                   'GOOGLE_VERTEX_BASE_URL', 'GOOGLE_GEMINI_BASE_URL']},
                               'dependencies': {p: version(p) for p in ['openai', 'google-genai', 'numpy', 'pandas']}},
                              sort_keys=True)
        journal = RequestJournal(path)
        try:
            journal.db.execute('CREATE TABLE IF NOT EXISTS metadata (identity TEXT NOT NULL)')
            saved = journal.db.execute('SELECT identity FROM metadata').fetchone()
            if args.resume:
                if saved is None or saved[0] != identity:
                    raise ValueError('Checkpoint configuration, inputs, code or dependencies changed')
            else:
                journal.db.execute('INSERT INTO metadata VALUES (?)', (identity,))
                journal.db.commit()
            # Candidate shuffling must repeat when reconstructing the search tree.
            import numpy as np
            journal.random.seed(args.seed)
            np.random.seed(args.seed)
            yield journal
        finally:
            journal.close()
