#!/usr/bin/env python3
"""Attribute resident memory to mapped files without reading model contents."""
import argparse
import fcntl
import os
from pathlib import Path
import re
import signal
import time

from collect_runtime_logs import cgroup_path, emit, logger, read
from runtime_lifecycle import collector_heartbeat, identity, runtime_log_root

HEADER = re.compile(r'^[0-9a-f]+-[0-9a-f]+\s')
FIELDS = ('Size', 'Rss', 'Pss', 'Anonymous', 'Private_Dirty', 'Shared_Dirty', 'Swap')
MODEL_SUFFIXES = {'.safetensors', '.gguf', '.ckpt', '.pth', '.pt', '.bin', '.onnx'}


def mapped_files(lines):
    files = {}
    anonymous_rss = 0
    current = None
    values = {}

    def flush():
        nonlocal anonymous_rss
        if current is None:
            return
        device, inode, path = current
        if not path.startswith('/') or inode == '0':
            anonymous_rss += values.get('Rss', 0)
            return
        key = (device, inode, path)
        row = files.setdefault(key, dict(path=path, device=device, inode=inode,
                                        mappings=0, **{name.lower()+'_bytes': 0 for name in FIELDS}))
        row['mappings'] += 1
        for name in FIELDS:
            row[name.lower()+'_bytes'] += values.get(name, 0)

    for line in lines:
        if HEADER.match(line):
            flush()
            parts = line.rstrip('\n').split(None, 5)
            current = (parts[3], parts[4], parts[5] if len(parts) == 6 else '')
            values = {}
        elif ':' in line:
            name, value = line.split(':', 1)
            if name in FIELDS:
                values[name] = int(value.split()[0])*1024
    flush()
    result = sorted(files.values(), key=lambda r: r['rss_bytes'], reverse=True)
    for row in result:
        row['file_backed_rss_bytes'] = max(0, row['rss_bytes']-row['anonymous_bytes'])
    return result, anonymous_rss


def snapshot(pid):
    path = Path('/proc') / str(pid)
    # Process start ticks distinguish reused PIDs.
    before = (path/'stat').read_text().rsplit(')', 1)[1].split()[19]
    with (path/'smaps').open() as stream:
        rows, anonymous = mapped_files(stream)
    after = (path/'stat').read_text().rsplit(')', 1)[1].split()[19]
    if before != after:
        raise ProcessLookupError('PID changed during smaps read')
    kept = [r for r in rows if r['rss_bytes'] >= 1024**2
            or Path(r['path'].removesuffix(' (deleted)')).suffix.lower() in MODEL_SUFFIXES]
    kept_keys = {(r['device'],r['inode'],r['path']) for r in kept}
    omitted = [r for r in rows if (r['device'],r['inode'],r['path']) not in kept_keys]
    return dict(pid=pid, process_start_ticks=before, files=kept,
                mapped_file_rss_bytes=sum(r['file_backed_rss_bytes'] for r in rows),
                mapped_files_pss_bytes=sum(r['pss_bytes'] for r in rows),
                anonymous_mapping_rss_bytes=anonymous, omitted_file_count=len(omitted),
                omitted_file_rss_bytes=sum(r['file_backed_rss_bytes'] for r in omitted))


def comfy_pids(comfyui_dir):
    main = os.fsencode(str(Path(comfyui_dir)/'main.py'))
    for path in Path('/proc').glob('[0-9]*'):
        try:
            args = (path/'cmdline').read_bytes().split(b'\0')
            if len(args)>1 and args[1] == main:
                yield int(path.name)
        except OSError:
            continue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', default=str(runtime_log_root()))
    parser.add_argument('--comfyui-dir', default='/storage/ComfyUI')
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval < 1:parser.error('interval must be >=1')
    root = Path(args.log_dir);root.mkdir(parents=True,exist_ok=True)
    lock = (root/'file-memory.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    log = logger(root,'file_memory')
    stopping=[]
    def stop(sig,frame):stopping.append(sig)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
    emit(log,event='file_memory_start',pid=os.getpid(),identity=identity(),interval=args.interval,
         scope='ComfyUI file mappings; unmapped page cache is not attributed; RSS may count shared pages more than once')
    while not stopping:
        started=time.monotonic()
        pids=list(comfy_pids(args.comfyui_dir))
        records=[]
        for pid in pids:
            try:records.append(snapshot(pid))
            except OSError as error:records.append(dict(pid=pid,error_type=type(error).__name__))
        stat = dict(line.split() for line in (read(cgroup_path()/'memory.stat') or '').splitlines())
        emit(log,event='file_memory_sample',processes=records,
             status='sampled' if pids else 'no_comfyui_process',
             cgroup_file_bytes=int(stat.get('file',0)),cgroup_file_mapped_bytes=int(stat.get('file_mapped',0)),
             sampling_seconds=round(time.monotonic()-started,4))
        for handler in log.handlers:
            handler.flush();os.fsync(handler.stream.fileno())
        collector_heartbeat(root, 'file-memory', active=True,
                            sampled_pids=[r['pid'] for r in records if 'error_type' not in r])
        if args.once:return
        while not stopping and time.monotonic()-started < args.interval:time.sleep(.1)


if __name__=='__main__':main()
