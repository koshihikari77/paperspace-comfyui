"""Read only the current pod's status/events using its mounted service account.

No permission changes, credential logging, or Core Machine API substitution.
"""
import json
import os
from pathlib import Path
import ssl
import urllib.error
import urllib.parse
import urllib.request

from runtime_lifecycle import identity, utc_now, write_state

SERVICE_ACCOUNT = Path('/var/run/secrets/kubernetes.io/serviceaccount')


def get_json(path):
    host = os.environ.get('KUBERNETES_SERVICE_HOST')
    if not host or not (SERVICE_ACCOUNT/'token').exists():
        return None, {'status': 'credentials_unavailable'}
    token = (SERVICE_ACCOUNT/'token').read_text().strip()
    port = os.environ.get('KUBERNETES_SERVICE_PORT_HTTPS', '443')
    request = urllib.request.Request(f'https://{host}:{port}{path}',
                                     headers={'Authorization': 'Bearer '+token})
    context = ssl.create_default_context(cafile=str(SERVICE_ACCOUNT/'ca.crt'))
    # Never forward the mounted credential to a redirect target.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context), NoRedirect())
    try:
        with opener.open(request, timeout=2) as response:
            return json.load(response), {'status': 'readable', 'http_status': response.status}
    except urllib.error.HTTPError as error:
        return None, {'status': 'denied' if error.code in (401, 403) else 'http_error',
                      'http_status': error.code}
    except (OSError, ValueError) as error:
        return None, {'status': 'read_failed', 'error_type': type(error).__name__}


def observe():
    result = dict(time=utc_now(), identity=identity(),
                  provider_shutdown_history={'status': 'not_connected'},
                  provider_auto_shutdown={'status': 'not_connected'})
    try:
        namespace = (SERVICE_ACCOUNT/'namespace').read_text().strip()
    except OSError:
        result['pod_status'] = {'status': 'credentials_unavailable'}
        return result
    name = os.environ.get('HOSTNAME', '')
    prefix = '/api/v1/namespaces/'+urllib.parse.quote(namespace, safe='')
    pod, result['pod_status'] = get_json(prefix+'/pods/'+urllib.parse.quote(name, safe=''))
    if pod:
        status = pod.get('status', {})
        result['pod_status']['observation'] = {
            key: status[key] for key in ('phase', 'reason', 'startTime') if key in status}
        result['pod_status']['containers'] = []
        for row in status.get('containerStatuses', []):
            container = {key: row[key] for key in ('name', 'restartCount') if key in row}
            for field in ('state', 'lastState'):
                container[field] = {
                    kind: {key: value[key] for key in ('reason', 'exitCode', 'signal', 'startedAt', 'finishedAt') if key in value}
                    for kind, value in row.get(field, {}).items()}
            result['pod_status']['containers'].append(container)
        uid = pod.get('metadata', {}).get('uid')
        if uid:
            events, result['pod_events'] = get_json(prefix+'/events?'+urllib.parse.urlencode(
                {'fieldSelector': 'involvedObject.uid='+uid, 'limit': 100}))
            if events:
                # Reasons/types/timestamps only; event messages may include workload content.
                result['pod_events']['observations'] = [
                    {key: row[key] for key in ('reason', 'type', 'count', 'firstTimestamp', 'lastTimestamp') if key in row}
                    for row in events.get('items', [])]
    else:
        result['pod_events'] = {'status': 'pod_identity_unavailable'}
    return result


def save(root):
    result = observe()
    write_state(Path(root)/'platform-state.json', result)
    return result
