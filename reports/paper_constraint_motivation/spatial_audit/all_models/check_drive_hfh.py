"""Measure the stored HFH MedSAM3 native baseline using the common diagnostics."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT),str(HERE.parent)]
from spatial_metrics import counts


def main(path):
    with zipfile.ZipFile(path) as z:
        checks=json.loads(z.read('manifest.json'))
        assert all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in checks.items())
        audit=json.loads(z.read('analysis.json'))
        assert audit['all_prediction_hashes_verified']
        rows=[]
        for rec in audit['records']:
            if rec['split']!='validation':
                continue
            case=rec['case']
            with np.load(io.BytesIO(z.read('baseline__'+case+'.npz'))) as f:
                truth=f['truth'].transpose(0,2,1).copy()
                pred=f['prediction'].transpose(0,2,1).copy()
            orig=ROOT/'reports/hfh_colab_exploration_20261111/baseline_v2/bundle/data'/(case+'.npz')
            with np.load(orig) as f:
                assert np.array_equal(truth,f['label'])
            r=counts(truth,pred,np.diag([1.,2.,1.,1.]),2)
            assert abs(r['dice']-rec['baseline']['dice'])<1e-10
            rows.append(dict(dataset='HFH',fold=0,seed=0,step=125,case=case,**r))
        assert len(rows)==10
        with (HERE/'drive_hfh_medsam3_f0s0_u125.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        print(len(rows),'recovered masks checked')


if __name__=='__main__':
    main(Path(sys.argv[1]))
