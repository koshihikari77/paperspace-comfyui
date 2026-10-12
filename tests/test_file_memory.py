import mmap
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from collect_file_memory import mapped_files,snapshot


class FileMemoryTests(unittest.TestCase):
    def test_aggregate_segments_preserve_spaces_and_anonymous_copy(self):
        text='''1000-2000 r--p 00000000 00:01 7 /models/Wan High.safetensors
Size: 4 kB
Rss: 4 kB
Pss: 2 kB
Anonymous: 0 kB
2000-3000 rw-p 00001000 00:01 7 /models/Wan High.safetensors
Size: 4 kB
Rss: 4 kB
Pss: 4 kB
Anonymous: 4 kB
3000-4000 rw-p 00000000 00:00 0 [heap]
Size: 4 kB
Rss: 4 kB
Pss: 4 kB
Anonymous: 4 kB
'''
        rows,anon=mapped_files(text.splitlines())
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['path'],'/models/Wan High.safetensors')
        self.assertEqual(rows[0]['rss_bytes'],8192)
        self.assertEqual(rows[0]['pss_bytes'],6144)
        self.assertEqual(rows[0]['file_backed_rss_bytes'],4096)
        self.assertEqual(anon,4096)

    def test_live_mmap_residency(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test model.safetensors'
            with path.open('w+b') as file:
                file.truncate(2*1024**2)
                with mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ) as mapping:
                    for offset in range(0,len(mapping),4096):value=mapping[offset]
                    result=snapshot(os.getpid())
                    row=next(r for r in result['files'] if r['path']==str(path))
                    self.assertGreaterEqual(row['file_backed_rss_bytes'],1024**2)
                    self.assertGreater(row['pss_bytes'],0)
                    self.assertEqual(row['anonymous_bytes'],0)


if __name__=='__main__':unittest.main()
