"""Build scheduler-only upgrade without changing scientific worker sources."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

from scripts.colab_hfh import build_medsam_constraints as base


def main():
    asset=base.REPORT/'hfh_medsam_constraints_v1.zip'
    with zipfile.ZipFile(asset) as archive:
        previous={name:archive.read(name) for name in archive.namelist() if name.startswith('source/')}
    base.main()
    with zipfile.ZipFile(asset) as archive:
        immutable={name:hashlib.sha256(data).hexdigest() for name,data in previous.items()
                   if not name.endswith(('medsam_constraints_notebook.py','medsam_constraints_parallel.py','medsam_constraints_capacity.py','medsam_constraints_drive.py'))}
        for name,expected in immutable.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=expected:
                raise ValueError('Scientific source changed during scheduler upgrade: '+name)
    notebook_path=base.REPORT/'HFH_MedSAM3_200_3fold_3seed.ipynb'
    notebook=json.loads(notebook_path.read_text())
    launcher=''.join(notebook['cells'][4]['source'])
    old="    if 'scripts.colab_hfh.medsam_constraints_campaign' in (p.info['cmdline'] or []): active.append(p)"
    new="    if any(name in (p.info['cmdline'] or []) for name in ('scripts.colab_hfh.medsam_constraints_campaign','scripts.colab_hfh.medsam_constraints_parallel')): active.append(p)"
    assert launcher.count(old)==1
    launcher=launcher.replace(old,new)
    launcher=launcher.replace("if active and '--strict' not in active[0].cmdline():", "if active and 'scripts.colab_hfh.medsam_constraints_parallel' not in active[0].cmdline():")
    launcher=launcher.replace("CONSTRAINT_SESSION_HOURS=20","CONSTRAINT_SESSION_HOURS=18")
    launcher=launcher.replace("[str(PYTHON),'-m','scripts.colab_hfh.medsam_constraints_campaign',", "[str(PYTHON),'-m','scripts.colab_hfh.medsam_constraints_parallel',")
    launcher=launcher.replace("'--root',str(ROOT),'--hours',str(CONSTRAINT_SESSION_HOURS),'--strict']", "'--root',str(ROOT),'--hours',str(CONSTRAINT_SESSION_HOURS),'--max-workers','0']")
    ast.parse(launcher)
    notebook['cells'][4]['source']=launcher
    notebook['cells'][0]['source'] += '''

## Adaptive GPU concurrency

This scheduler upgrade preserves all completed archives and the exact scientific worker sources. It benchmarks real pending experiments at one, two and higher worker counts only when measured GPU/host memory and CPU reserves permit. Training throughput uses stable 180-second windows; multi-worker levels require two windows. Extra workers must improve aggregate updates/minute by at least8% over one worker. The fastest measured safe count is retained; a reduced target drains existing workers without discarding their updates. Each constraint arm is measured separately because calibration and memory demands differ.

The notebook prints each worker's task and saves live GPU utilization, memory, progress and concurrency measurements in `monitor/live_status.json` and `monitor/parallel_benchmark.json`. Fail-fast BAND stops new launches and preserves any in-flight recovery. The nine-pair qualification criterion and full-matrix definitions are unchanged. Sessions are capped at18hours after this handoff. Run cell4 only after the old serial controller has finished its active run and exited cleanly; do not rerun setup.
'''
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    notebook_path.write_text(json.dumps(notebook,indent=1)+'\n')
    (base.REPORT/'medsam_constraints_launcher.py').write_text(launcher)
    receipt=dict(scientific_sources_unchanged=immutable,overlay_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
                 scheduler='adaptive concurrency',session_hours=18)
    (base.REPORT/'medsam_parallel_upgrade.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(overlay_sha256=receipt['overlay_sha256'],unchanged_scientific_files=len(immutable),notebook=str(notebook_path))))


if __name__=='__main__':main()
