import json
import pickle
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from Method.checkpoint import RequestJournal, checkpoint_run, RequestInterrupted
from types import SimpleNamespace


# Separate Python processes rebuild the native five-level branching tree. The fake
# search line uses the real SDK wrapper; only remote generation/recombination is mocked.
SEARCH_PROGRAM = r'''
import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from openai.types.chat import ChatCompletion
from Method.checkpoint import RequestJournal, _scope
from Method.utils import llm_generation
from Method.hierarchy_greedy import HierarchyGreedy

out = Path(sys.argv[1]); crash = sys.argv[2] == 'crash'
(out / 'input.json').write_text(json.dumps([['question', 'background', 'coarse']]))
network = out / 'network.jsonl'
def create(**request):
    identity = f'{_scope.get().path}:{_scope.get().calls}'
    with network.open('a') as log: log.write(json.dumps({'id': identity, 'request': request}) + '\n')
    if crash and identity == 'root/pool0/task1/pool0/task1:1':
        time.sleep(.15)
        os._exit(23)
    # Reverse relative worker speeds when rebuilding; scope and aggregation must not change.
    time.sleep(.001 if ('task0' in identity) == crash else .004)
    return ChatCompletion(id=identity, object='chat.completion', created=0, model='test',
        choices=[{'index': 0, 'message': {'role': 'assistant', 'content': identity}, 'finish_reason': 'stop'}])
fake = SimpleNamespace(base_url='https://example.test/v1/', chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
journal = RequestJournal(out / 'requests.sqlite3')
client = fake
search = HierarchyGreedy.__new__(HierarchyGreedy)
search.args = SimpleNamespace(if_use_custom_research_background_and_coarse_hyp=1,
    custom_research_background_and_coarse_hyp_path=str(out/'input.json'), num_hierarchy=5,
    beam_size_branching=2, if_parallel=1, num_init_for_EU=2, num_recom_trial_for_better_hyp=1,
    if_save=1, output_dir=str(out/'tree.pkl'), beam_compare_mode=0)
search.pairwise_compare = None
def line(q, survey, coarse, hierarchy, branch, init, previous, prompt_type):
    # Two identical samples must remain distinct, even inside identical parallel branches.
    request = {'model': 'test', 'messages': [{'role': 'user', 'content': survey}]}
    first = llm_generation(survey, 'test', client)
    second = llm_generation(survey, 'test', client)
    return [[first + '+' + second, 'reason']]
search.search_along_one_search_line = line
search.recombination_from_local_minimums = lambda q,s,c,h,lines,b: ([], lines[0][-1][0], 0)
search.get_finegrained_hyp_for_one_research_question_Branching(0)

journal.close()
'''


CLI_PROGRAM = r'''
import json, os, runpy, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from openai.types.chat import ChatCompletion
out = Path(sys.argv[1]); mode = sys.argv[2]
network = out / 'network.jsonl'
count = 0
def create(**request):
    global count
    count += 1
    with network.open('a') as f: f.write(json.dumps(request) + '\n')
    if mode == 'crash' and count == 2:
        os._exit(23)
    content = ('Reasoning Process starts: reason Reasoning Process ends\n'
               'Revised Hypothesis starts: proposal Revised Hypothesis ends\n'
               'Selection of research hypothesis candidate starts: 2 Selection of research hypothesis candidate ends')
    return ChatCompletion(id='test', object='chat.completion', created=0, model='test',
        choices=[{'index':0, 'message':{'role':'assistant','content':content}, 'finish_reason':'stop'}])
client = SimpleNamespace(base_url='https://example.test', chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
sys.argv = ['Method/hierarchy_greedy.py', '--api_key', 'unused', '--eval_api_key', 'unused',
            '--model_name', 'test', '--eval_model_name', 'test', '--base_url', 'https://example.test',
            '--if_use_custom_research_background_and_coarse_hyp', '1',
            '--custom_research_background_and_coarse_hyp_path', str(out/'input.json'),
            '--output_dir', str(out/'Checkpoints/tree.pkl'), '--if_save', '1', '--max_search_step', '2',
            '--num_init_for_EU', '1', '--num_hierarchy', '1', '--checkpoint', str(out/'checkpoint.sqlite3')]
if mode != 'crash': sys.argv += ['--resume']
with patch('openai.OpenAI', return_value=client):
    runpy.run_path('Method/hierarchy_greedy.py', run_name='__main__')
'''


class CheckpointTest(unittest.TestCase):
    def test_process_crash_replays_successes_and_resends_only_missing_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resumed, reference = root/'resumed', root/'reference'
            resumed.mkdir(); reference.mkdir()
            env = dict(os.environ)
            def launch(path, mode):
                return subprocess.run([sys.executable, '-c', SEARCH_PROGRAM, str(path), mode], env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            first = launch(resumed, 'crash')
            self.assertEqual(first.returncode, 23, first.stderr)
            db = sqlite3.connect(resumed/'requests.sqlite3')
            cached = {f'{scope}:{ordinal}' for scope, ordinal in db.execute('SELECT scope, ordinal FROM calls WHERE response IS NOT NULL')}
            self.assertTrue(cached)
            before = [json.loads(line) for line in (resumed/'network.jsonl').read_text().splitlines()]
            second = launch(resumed, 'continue')
            self.assertEqual(second.returncode, 0, second.stderr)
            completed = launch(reference, 'continue')
            self.assertEqual(completed.returncode, 0, completed.stderr)
            def normalized(value):
                if isinstance(value, dict):
                    return {k: normalized(v) for k, v in value.items() if k != 'parent'}
                if isinstance(value, list):
                    return [normalized(v) for v in value]
                return value
            self.assertEqual(normalized(pickle.loads((resumed/'tree.pkl').read_bytes())),
                             normalized(pickle.loads((reference/'tree.pkl').read_bytes())))
            after = [json.loads(line) for line in (resumed/'network.jsonl').read_text().splitlines()][len(before):]
            self.assertFalse(cached & {row['id'] for row in after})
            self.assertIn('root/pool0/task1/pool0/task1:1', {row['id'] for row in after})
            successful = db.execute('SELECT count(*) FROM calls WHERE response IS NOT NULL').fetchone()[0]
            reference_calls = (reference/'network.jsonl').read_text().splitlines()
            self.assertEqual(successful, len(reference_calls))
            self.assertEqual(successful, 40)  # 5 levels x 2 branches x 2 initializations x 2 samples
            db.close()

    def test_request_mismatch_rejected_without_sending(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'journal.sqlite3'
            j = RequestJournal(p); j.call({'model':'test','prompt':'first'}, lambda: {'answer':1}); j.close()
            j = RequestJournal(p)
            try:
                with self.assertRaisesRegex(ValueError, 'request mismatch'):
                    j.call({'model':'other','prompt':'first'}, lambda: self.fail('sent changed request'))
            finally: j.close()

    def test_in_process_retry_does_not_advance_failed_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            j = RequestJournal(Path(directory)/'journal.sqlite3')
            try:
                with self.assertRaises(ConnectionError):
                    j.call({'x':1}, lambda: (_ for _ in ()).throw(ConnectionError()))
                self.assertEqual(j.call({'x':1}, lambda: {'value':2}), {'value':2})
                self.assertEqual(j.call({'x':1}, lambda: {'value':3}), {'value':3})
            finally: j.close()
            j = RequestJournal(Path(directory)/'journal.sqlite3')
            try:
                self.assertEqual(j.call({'x':1}, lambda: self.fail('unexpected API')), {'value':2})
                self.assertEqual(j.call({'x':1}, lambda: self.fail('unexpected API')), {'value':3})
            finally: j.close()

    def test_native_cli_crash_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/'input.json').write_text(json.dumps([['question', 'background', 'coarse']]))
            (out/'Checkpoints').mkdir()
            def launch(mode):
                return subprocess.run([sys.executable, '-c', CLI_PROGRAM, str(out), mode], capture_output=True, text=True)
            first = launch('crash')
            self.assertEqual(first.returncode, 23, first.stderr)
            before = (out/'network.jsonl').read_text().splitlines()
            # A partial final artifact must not make --resume skip the search.
            (out/'Checkpoints/tree.pkl').write_bytes(b'partial')
            second = launch('resume')
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIsInstance(pickle.loads((out/'Checkpoints/tree.pkl').read_bytes()), dict)
            after = (out/'network.jsonl').read_text().splitlines()[len(before):]
            self.assertNotIn(before[0], after)
            self.assertIn(before[1], after)

    def args(self, root, resume=False):
        source = root / 'input.txt'
        if not source.exists():
            source.write_text('question')
        return SimpleNamespace(checkpoint=str(root/'journal.sqlite3'), resume=resume,
                               api_key='secret', eval_api_key='secret2', seed=17,
                               if_use_custom_research_background_and_coarse_hyp=1,
                               custom_research_background_and_coarse_hyp_path=str(source))

    def test_run_lock_rejects_second_process(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            with checkpoint_run(args):
                code = ('from Method.checkpoint import checkpoint_run; from types import SimpleNamespace; '
                        'import json,sys\nwith checkpoint_run(SimpleNamespace(**json.loads(sys.argv[1]))): pass')
                result = subprocess.run([sys.executable, '-c', code, json.dumps(vars(args))], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('already in use', result.stderr)

    def test_resume_validates_inputs_config_and_preserves_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args = self.args(root)
            with checkpoint_run(args):
                pass
            args.resume = True
            args.api_key = 'replacement'
            with checkpoint_run(args):
                pass
            with sqlite3.connect(args.checkpoint) as db:
                identity = db.execute('select identity from metadata').fetchone()[0]
                self.assertNotIn('secret', identity)
            args.seed = 99
            with self.assertRaisesRegex(ValueError, 'changed'):
                with checkpoint_run(args): pass
            args.seed = 17
            Path(args.custom_research_background_and_coarse_hyp_path).write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                with checkpoint_run(args): pass

    def test_raw_response_is_saved_before_text_extraction(self):
        from Method.utils import llm_generation
        from openai.types.chat import ChatCompletion
        response = ChatCompletion(id='test', object='chat.completion', created=0, model='test', choices=[])
        with tempfile.TemporaryDirectory() as directory:
            for iteration in range(2):
                with patch('time.sleep'):
                    create = unittest.mock.Mock(return_value=response)
                    if iteration:
                        create.side_effect = AssertionError('successful call was resent')
                    client = SimpleNamespace(base_url='https://example.test', chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
                    journal = RequestJournal(Path(directory)/'journal.sqlite3')
                    try:
                        with self.assertRaisesRegex(Exception, 'list index out of range'):
                            llm_generation('question', 'model', client)
                    finally:
                        journal.close()
                    self.assertEqual(create.call_count, 1 if iteration == 0 else 0)

    def test_failed_api_request_escapes_extraction_retry_loop(self):
        from Method.utils import llm_generation_with_extraction
        create = unittest.mock.Mock(side_effect=ConnectionError('offline'))
        client = SimpleNamespace(base_url='https://example.test', chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with tempfile.TemporaryDirectory() as directory:
            journal = RequestJournal(Path(directory)/'journal.sqlite3')
            try:
                with self.assertRaises(RequestInterrupted):
                    llm_generation_with_extraction('question', 'model', client, [('answer', 'text')])
                self.assertEqual(create.call_count, 1)
            finally:
                journal.close()


    def test_candidate_shuffle_is_independent_of_sdk_randomness(self):
        import random
        from Method.checkpoint import shuffle_candidates
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            first = list(range(30))
            second = list(range(30))
            with checkpoint_run(args):
                # SDK retry backoff can consume arbitrary global random draws.
                for _ in range(17): random.random()
                shuffle_candidates(first)
            args.resume = True
            with checkpoint_run(args):
                shuffle_candidates(second)
            self.assertEqual(first, second)

    def test_google_response_replay_and_no_checkpoint_calls(self):
        from Method.utils import llm_generation
        from google.genai import types
        response = types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(parts=[types.Part(text='answer')]))])
        create = unittest.mock.Mock(return_value=response)
        client = SimpleNamespace(models=SimpleNamespace(generate_content=create))
        with tempfile.TemporaryDirectory() as directory:
            for _ in range(2):
                journal = RequestJournal(Path(directory)/'journal.sqlite3')
                try:
                    self.assertEqual(llm_generation('question', 'model', client, api_type=2), 'answer')
                finally:
                    journal.close()
            self.assertEqual(create.call_count, 1)
            self.assertEqual(llm_generation('question', 'model', client, api_type=2), 'answer')
            self.assertEqual(create.call_count, 2)


if __name__ == '__main__': unittest.main()
