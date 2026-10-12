import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from collect_gpu_events import event_fields
from diagnose_runtime import classify


class GPUEvidenceTests(unittest.TestCase):
    def test_xid_code_preserved(self):
        event=SimpleNamespace(eventType=8,eventData=79,gpuInstanceId=0xffffffff,computeInstanceId=0xffffffff)
        self.assertEqual(event_fields(event)['xid'],79)

    def test_xid_is_evidence_not_invented_shutdown_reason(self):
        result=classify([], [dict(event='gpu_driver_event',time='2026-09-26T00:00:00Z',xid=79)], [])
        self.assertEqual(result[0]['kind'],'gpu_driver_error')
        self.assertNotIn('shutdown_reason',result[0])

    def test_old_oom_count_not_new_oom(self):
        samples=[dict(cgroups=[{'path':'container','memory.events':'oom_kill 3'}])]*2
        self.assertEqual(classify([],[],samples),[])
        samples.append(dict(cgroups=[{'path':'container','memory.events':'oom_kill 4'}]))
        self.assertEqual(classify([],[],samples)[0]['delta'],1)

    def test_no_evidence_is_not_classified_as_platform_shutdown(self):
        self.assertEqual(classify([],[],[]),[])


if __name__=='__main__':unittest.main()
