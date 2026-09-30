"""Post-hoc sensitivity to the near-constant closing-based solidity proxy.

Motivated by the case033 validation failure; never an untouched validation
result. Preserve the original audit and use identical splits/hyperparameters.
"""
import json
import numpy as np
from sklearn.model_selection import KFold
from .audit_cue_consistency import ROOT, fit_predict, metrics


def main():
    out=ROOT/'experiments/uncal_cue_consistency_20260921'
    cs=json.loads((out/'features.json').read_text())
    train=[c for c in cs if c['split']=='train'];val=[c for c in cs if c['split']=='val']
    cols=[i for i,n in enumerate(cs[0]['names']) if 'solidity_proxy' not in n]
    oof=np.zeros(len(train),int)
    for tr,te in KFold(4,shuffle=True,random_state=0).split(train):
        p,_=fit_predict([train[i] for i in tr],[train[i] for i in te],cols,True)
        oof[te]=p
    pred,fit=fit_predict(train,val,cols,True)
    result=dict(reason='Post-hoc sensitivity after inspecting validation case033; exclude current/delta/next-delta of near-constant closing-based solidity proxy.',
                removed=[n for n in cs[0]['names'] if 'solidity_proxy' in n],
                train_oof=metrics(train,oof),val=metrics(val,pred),fit=fit,
                predictions=[dict(name=c['name'],split=c['split'],target=c['target'],pred=int(p)) for c,p in zip(train+val,list(oof)+pred)])
    (out/'sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('fit','predictions')},indent=2))


if __name__=='__main__':main()
