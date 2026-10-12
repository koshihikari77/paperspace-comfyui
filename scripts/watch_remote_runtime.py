#!/usr/bin/env python3
"""Run on another computer: keep independent evidence over existing SSH access."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from collect_runtime_logs import emit, logger


def observe(target, notebook):
    root = '/notebooks/logs/runtime/notebooks/' + notebook
    command = 'cat ' + shlex.quote(root + '/lifecycle-state.json')
    try:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                                 target, command], capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(status='ssh_unavailable', error_type=type(error).__name__)
    if result.returncode:
        return dict(status='ssh_or_remote_read_failed', returncode=result.returncode)
    try:
        data = json.loads(result.stdout)
        last = dt.datetime.fromisoformat(data['last_seen'].replace('Z','+00:00'))
        age = (dt.datetime.now(dt.timezone.utc)-last).total_seconds()
        return dict(status='heartbeat_stale' if age > 45 else 'heartbeat_recent',
                    heartbeat_age_seconds=round(age,1), session=data,
                    note='SSH failure or stale heartbeat alone does not prove provider shutdown')
    except (ValueError, KeyError, TypeError):
        return dict(status='invalid_remote_state')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh-target', required=True, help='Existing SSH config alias or user@host')
    parser.add_argument('--notebook-id', required=True)
    parser.add_argument('--log-dir', default='./paperspace-external-logs')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.ssh_target.startswith('-') or not re.fullmatch(r'[a-zA-Z0-9_-]+',args.notebook_id):
        parser.error('invalid SSH target or notebook ID')
    if os.environ.get('PAPERSPACE_NOTEBOOK_ID')==args.notebook_id:
        parser.error('Run this observer OUTSIDE the monitored notebook')
    root=Path(args.log_dir);root.mkdir(parents=True,exist_ok=True)
    log=logger(root,'external')
    try:
        while True:
            emit(log,event='remote_observation',notebook_id=args.notebook_id,
                 **observe(args.ssh_target,args.notebook_id))
            for handler in log.handlers:
                handler.flush();os.fsync(handler.stream.fileno())
            if args.once:return
            time.sleep(10)
    except KeyboardInterrupt:
        return


if __name__=='__main__':main()
