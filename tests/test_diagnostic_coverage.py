import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import check_crash_diagnostics as checks
from runtime_lifecycle import identity


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.node = self.root/'custom_nodes/runtime_diagnostics'
        self.node.mkdir(parents=True)
        hashes = {}
        for name in ('__init__.py', 'memory_debug.py', 'tensor_retention.py'):
            (self.node/name).write_text('revision one')
            hashes[name] = hashlib.sha256(b'revision one').hexdigest()
        data = {'lifecycle-state': dict(identity(), pid=1),
                'supervisor-state': dict(identity=identity(), status='running', pid=2, supervisor_pid=1),
                'hooks-state': dict(pid=2, boot_id=identity()['boot_id'], process_start_ticks='10',
                                    node_hooks_loaded=True, tensor_retention_hooks_loaded=True, source_hashes=hashes),
                'gpu-events-state': dict(active=True, identity=identity(), pid=3),
                'mode': dict(mode='debug')}
        for name in ('fast-state', 'file-memory-state'):
            data[name] = dict(active=True, identity=identity(), pid=4, process_start_ticks='10', sampled_pids=[2])
        for name, state in data.items():
            (self.root/(name+'.json')).write_text(json.dumps(state))
        for name, value in [('process_matches', True), ('process_start_ticks', '10')]:
            patcher = patch.object(checks, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def report(self):
        return checks.coverage(self.root, self.root)

    def test_verified_local_is_not_complete_cause_coverage(self):
        result = self.report()
        self.assertTrue(result['local_ready'], result)
        self.assertFalse(result['complete_cause_coverage'])

    def test_missing_retention_fails_readiness(self):
        path = self.root/'hooks-state.json'
        state = json.loads(path.read_text())
        state['tensor_retention_hooks_loaded'] = False
        path.write_text(json.dumps(state))
        self.assertIn('tensor_retention_hooks_loaded', self.report()['local_missing'])
        self.assertFalse(checks.verify(self.root, self.root)['local_ready'])

    def test_edited_but_not_loaded_code_fails(self):
        (self.node/'tensor_retention.py').write_text('revision two')
        self.assertIn('loaded_sources_current', self.report()['local_missing'])

    def test_reused_pid_fails_even_with_fresh_log(self):
        with patch.object(checks, 'process_start_ticks', return_value='11'):
            result = self.report()
        self.assertFalse(result['node_hooks_loaded'])
        self.assertFalse(result['one_second_memory'])

    def test_dead_collector_fails_even_with_fresh_state(self):
        with patch.object(checks, 'process_matches', side_effect=lambda pid, script: script != 'collect_fast_runtime.py'):
            self.assertFalse(self.report()['one_second_memory'])

    def test_rotated_logs_do_not_erase_loaded_hook_evidence(self):
        self.assertFalse((self.root/'nodes.jsonl').exists())
        self.assertTrue(self.report()['node_hooks_loaded'])


if __name__ == '__main__':
    unittest.main()
