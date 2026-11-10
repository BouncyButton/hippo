"""Freeze disjoint Colab/cluster assignments and an absolute Colab deadline."""
import ast
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'reports/sam3_geofwi_f013_s012_126_20261019/split_cluster_20261022'


def main():
    sys.path.insert(0, str(HERE))
    import replication
    OUT.mkdir(parents=True, exist_ok=True)
    state_bytes = Path('/tmp/sam3_split_state.json').read_bytes()
    state = json.loads(state_bytes)
    tasks = replication.tasks()
    pending = [t for t in tasks if t['id'] not in state['completed']]
    pending.sort(key=lambda t: (t['id'] not in state['charges'], tasks.index(t)))
    cutoff = datetime(2026, 10, 22, 15, tzinfo=ZoneInfo('Europe/Rome')).timestamp()
    plan = dict(version=1, total=126, completed_at_split=len(state['completed']),
                colab_tasks=[t['id'] for t in pending[:6]],
                cluster_tasks=[t['id'] for t in pending[6:]],
                deadline_epoch=cutoff, stop_compute_epoch=cutoff-300,
                source_state_id='1ro_AE2OwJI8jlEeKJGIjwR7kUoZ07tPX',
                source_state_sha256=hashlib.sha256(state_bytes).hexdigest(),
                policy='Disjoint ownership; Colab admission requires 45 minutes (15 for trained evaluation resumes); stop compute 14:55 Rome to save recovery before 15:00. Deferred Colab work must be reconciled after cutoff.')
    assert set(plan['colab_tasks']).isdisjoint(plan['cluster_tasks'])
    assert len(pending) + len(state['completed']) == 126
    (OUT/'assignment.json').write_text(json.dumps(plan, indent=2)+'\n')
    (OUT/'source_campaign_state.json').write_bytes(state_bytes)
    nb = json.loads(Path('/tmp/sam3_notebook_before_split.ipynb').read_text())
    (OUT/'notebook_before.ipynb').write_text(json.dumps(nb, indent=2)+'\n')
    boot = nb['cells'][1]
    source = ''.join(boot['source'])
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    edits = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == 'write_text':
                receiver = call.func.value
                if isinstance(receiver, ast.Name) and receiver.id == 'LAUNCHER':
                    edits.append((node.lineno-1, node.end_lineno,
                                  'LAUNCHER.write_text('+repr((HERE/'resume_other_account_20261021.py').read_text())+')\n'))
    if len(edits) != 1:
        raise ValueError('Could not locate the account launcher cell')
    for start, stop, text in reversed(edits):
        lines[start:stop] = [text]
    source = ''.join(lines)
    source += '\nSPLIT_ASSIGNMENT = '+repr(plan)+'\n'
    source += 'handoff=json.loads(LAUNCHER.with_suffix(".json").read_text())\n'
    source += 'handoff.update(colab_tasks=SPLIT_ASSIGNMENT["colab_tasks"], stop_compute_epoch=SPLIT_ASSIGNMENT["stop_compute_epoch"])\n'
    source += 'LAUNCHER.with_suffix(".json").write_text(json.dumps(handoff))\n'
    source += 'print("Colab owns",len(SPLIT_ASSIGNMENT["colab_tasks"]),"runs; all other pending runs belong to the cluster.")\n'
    boot['source'] = [source]
    run = ''.join(nb['cells'][5]['source'])
    run = run.replace('remaining=SESSION_HOURS-(time.monotonic()-STARTED)/3600',
        'remaining=(SPLIT_ASSIGNMENT["stop_compute_epoch"]-time.time())/3600')
    run = run.replace('Session budget exhausted; resume in a fresh matching runtime',
        'The 15:00 Rome cutoff has been reached; remaining work belongs to the cluster')
    nb['cells'][5]['source'] = [run]
    nb['cells'][0]['source'] = [
        '# SAM3: Colab until 15:00 Rome, 22 October 2026\n\n'
        '**Sign in as filippofocaccia2@gmail.com. Use A100 / High RAM and Run all.**\n\n'
        f'{len(state["completed"])}/126 runs are complete. This session owns the next six runs, including '
        'the interrupted evaluation. The other 47 pending runs are assigned exclusively to two cluster workers.\n\n'
        'A full run starts only with at least 45 minutes available (15 minutes for an update-25 evaluation resume). '
        'Computation stops at 14:55 to allow recovery uploads before 15:00. Anything deferred is handed to the cluster. '
        'Completed runs are never repeated. Do not launch the old unrestricted notebook.\n'
    ]
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []; cell['execution_count'] = None
            cell['metadata'].pop('executionInfo', None)
            ast.parse(''.join(cell['source']))
    (OUT/'SAM3_until_1500_cluster_remainder.ipynb').write_text(json.dumps(nb, indent=2)+'\n')
    print(json.dumps(dict(completed=len(state['completed']),colab=len(plan['colab_tasks']),cluster=len(plan['cluster_tasks'])),indent=2))


if __name__ == '__main__':
    main()
