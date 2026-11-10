"""Run unchanged frozen trainers, enforcing the reference initialization first."""
import sys
from pathlib import Path

bundle=Path(sys.argv.pop(1)).resolve()
sys.path.insert(0,str(bundle))
import worker

original=worker.configure


def configure(p, output):
    original(p,output)
    builder=worker.exp.base.build_model
    seed=0 if '--preflight' in sys.argv and p['model_identity']!='unetr_vit_b16' else p['seeds'][0]
    expected=p['reference_initializations'][str(seed)]
    def build(*args,**kwargs):
        model=builder(*args,**kwargs)
        if worker.io.model_digest(model)!=expected:
            raise ValueError('Initialization differs from established matching-seed baseline')
        return model
    worker.exp.base.build_model=build


worker.configure=configure
worker.main()
