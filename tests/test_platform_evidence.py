import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import platform_evidence as platform


class PlatformEvidenceTests(unittest.TestCase):
    def test_permission_denial_is_persisted_without_claiming_status(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, 'namespace').write_text('own-namespace')
            with patch.object(platform, 'SERVICE_ACCOUNT', Path(root)), patch.object(platform, 'get_json',
                    return_value=(None, {'status': 'denied', 'http_status': 403})):
                report = platform.save(root)
            self.assertEqual(report['pod_status']['http_status'], 403)
            self.assertEqual(report['provider_shutdown_history']['status'], 'not_connected')
            self.assertEqual(json.loads(Path(root, 'platform-state.json').read_text()), report)

    def test_termination_reason_kept_without_message_contents(self):
        pod = {'metadata': {'uid': 'own-uid'}, 'status': {'phase': 'Failed', 'reason': 'Evicted',
               'message': 'private workload content', 'containerStatuses': [{'name': 'notebook',
               'restartCount': 1, 'lastState': {'terminated': {'reason': 'OOMKilled', 'exitCode': 137,
                                                             'message': 'private workload content'}}}]}}
        with tempfile.TemporaryDirectory() as root:
            Path(root, 'namespace').write_text('own-namespace')
            with patch.object(platform, 'SERVICE_ACCOUNT', Path(root)), patch.object(platform, 'get_json',
                    side_effect=[(pod, {'status': 'readable'}), ({'items': []}, {'status': 'readable'})]):
                report = platform.observe()
        text = json.dumps(report)
        self.assertIn('OOMKilled', text)
        self.assertIn('Evicted', text)
        self.assertNotIn('private workload content', text)


if __name__ == '__main__':
    unittest.main()
