"""Persist collector lifecycle evidence; never infer a provider shutdown reason."""
import datetime as dt
import json
import os
from pathlib import Path
import signal
import uuid
import re
import tempfile


def runtime_log_root():
    notebook = os.environ.get('PAPERSPACE_NOTEBOOK_ID', 'local')
    notebook = re.sub(r'[^a-zA-Z0-9_-]', '_', notebook)[:128]
    return Path('/notebooks/logs/runtime/notebooks') / notebook


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def process_start_ticks(pid):
    try:
        return Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def collector_heartbeat(root, name, **fields):
    write_state(Path(root)/(name+'-state.json'), dict(identity=identity(), pid=os.getpid(),
                process_start_ticks=process_start_ticks(os.getpid()), last_seen=utc_now(), **fields))


def write_state(path, data):
    # Startup checks and the collector may update coverage concurrently.
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load_state(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def shutdown_setting(path, identity):
    """Only accept a manual observation tied to this notebook/container session."""
    data = load_state(path)
    if not isinstance(data, dict):
        return {'status': 'unknown', 'reason': 'no_valid_manual_observation'}
    if any(data.get(key) != identity.get(key) for key in ('notebook_id', 'container_id')):
        return {'status': 'unknown', 'reason': 'observation_from_different_session'}
    return {key: data[key] for key in ('status', 'source', 'observed_at', 'notebook_id',
                                      'container_id', 'shutdown_at') if key in data}


class Lifecycle:
    def __init__(self, root, emit, identity):
        self.root = Path(root)
        self.emit = emit
        self.path = self.root / 'lifecycle-state.json'
        self.identity = identity
        self.stop_signal = None
        self.handlers = {}
        previous = load_state(self.path)
        self.state = dict(identity, session_id=str(uuid.uuid4()), pid=os.getpid(),
                          started_at=utc_now(), last_seen=utc_now(), clean_exit=False)
        # The event is about the collector, not proof of container/VM failure.
        if isinstance(previous, dict):
            self.emit(event='previous_collector_session', previous=previous,
                      observation='clean_exit' if previous.get('clean_exit') else 'no_exit_record',
                      environment_changed=any(previous.get(k) != identity.get(k)
                                              for k in ('boot_id', 'container_id')),
                      shutdown_reason='unknown')
        self.emit(event='lifecycle_start', **self.state)
        self.setting = None
        self.heartbeat()

    def install_handlers(self):
        def stop(signum, frame):
            self.stop_signal = signum
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            self.handlers[sig] = signal.signal(sig, stop)

    def heartbeat(self):
        setting = shutdown_setting(self.root / 'shutdown-observation.json', self.identity)
        if setting != self.setting:
            self.emit(event='auto_shutdown_observation', observation=setting)
            self.setting = setting
        self.state.update(last_seen=utc_now(), auto_shutdown=self.setting)
        write_state(self.path, self.state)

    def close(self, reason):
        self.state.update(last_seen=utc_now(), clean_exit=True, exit_reason=reason)
        if self.stop_signal is not None:
            self.state['signal'] = signal.Signals(self.stop_signal).name
        self.emit(event='collector_exit', **self.state, shutdown_reason='unknown')
        write_state(self.path, self.state)
        for sig, handler in self.handlers.items():
            signal.signal(sig, handler)


def identity():
    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None
    # Paperspace reuses the notebook ID as hostname across container restarts.
    # The cgroup path includes the actual pod/container identity.
    cgroup = read('/proc/self/cgroup') or ''
    container = next((line.split(':', 2)[2] for line in cgroup.splitlines()
                      if line.startswith('0::')), None)
    if container == '/':
        container = None
    return dict(boot_id=read('/proc/sys/kernel/random/boot_id'),
                container_id=container,
                notebook_id=os.environ.get('PAPERSPACE_NOTEBOOK_ID'))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Record an observation from the Paperspace Notebook Instance panel.')
    parser.add_argument('--log-dir', default=str(runtime_log_root()))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--shutdown-at', help='Exact timestamp with timezone, e.g. 2026-09-27T01:00:00+09:00')
    group.add_argument('--disabled', action='store_true', help='Record that the console shows auto-shutdown disabled')
    args = parser.parse_args()
    data = dict(identity(), status='disabled' if args.disabled else 'scheduled',
                source='manual_console_observation', observed_at=utc_now())
    if args.shutdown_at:
        try:
            timestamp = dt.datetime.fromisoformat(args.shutdown_at.replace('Z', '+00:00'))
            if timestamp.utcoffset() is None:
                raise ValueError('timezone required')
            data['shutdown_at'] = timestamp.astimezone(dt.timezone.utc).isoformat()
        except ValueError:
            parser.error('--shutdown-at must be an ISO timestamp including timezone')
    root = Path(args.log_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_state(root / 'shutdown-observation.json', data)
    print('Saved manual observation. The collector will log it on its next heartbeat.')


if __name__ == '__main__':
    main()
