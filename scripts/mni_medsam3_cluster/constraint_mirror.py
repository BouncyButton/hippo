"""Optional durable archive mirror; scheduling and delivery remain on cluster."""
import json
from scripts.cross_dataset_campaign import collect

original_snapshot=collect.snapshot


def snapshot(stage):
    state=original_snapshot(stage)
    state.pop('base_request.json',None)
    code='''import json,sys
from pathlib import Path
p=Path(sys.argv[1]);print(json.dumps({n:json.loads((p/n).read_text()) for n in ('cohort.json','comparison.json','pilot_comparison.json','baseline_gate.json','service.json','service_error.json','geometry_preflight.json') if (p/n).exists()}))'''
    state['matrix_reports']=json.loads(collect.remote(['python3','-c',code,stage]))
    return state


if __name__=='__main__':
    collect.snapshot=snapshot
    collect.resume_after_archive=lambda *args:None
    collect.main()
