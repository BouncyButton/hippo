"""Instrument the frozen three-step GPU replay; does not launch production."""
import argparse, sys, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path);p.add_argument('--durable',type=Path);a=p.parse_args()
sys.path.insert(0,str(a.bundle))
import lr3_runtime as campaign
root=campaign.io.activate(a.bundle)
folder,protocol=campaign.protocol(root,'lr5_frame_bands')
output=root/'preflight_diagnostic';output.mkdir(exist_ok=True)
s=(root/'worker.py').read_text()
s=s.replace('first=step(0)', "first=step(0)\n    saved_model=io.model_digest(model)\n    saved_opt=deepcopy(optimizer.state_dict())")
s=s.replace('expected_opt=deepcopy(optimizer.state_dict());expected_rng=', "expected_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}\n        expected_opt=deepcopy(optimizer.state_dict());expected_rng=")
s=s.replace('actual_loss=step(1)', "restored_model=io.model_digest(model)\n        actual_loss=step(1)")
s=s.replace("if expected_loss!=actual_loss or expected!=io.model_digest(model):raise ValueError('Resume model/loss mismatch')", "model_exact=(expected==io.model_digest(model))")
start=s.index('        if (not equal(expected_opt,optimizer.state_dict())')
end=s.index('    evaluated=',start)
s=s[:start]+"""        differences=[]
        for k,v in model.state_dict().items():
            x=v.detach().cpu();y=expected_model[k]
            if not torch.equal(x,y):
                differences.append(dict(name=k,max_absolute=float((x-y).abs().max()),changed=int((x!=y).sum()),elements=x.numel()))
        report=dict(saved_model_restored_exactly=saved_model==restored_model,first_loss=first,expected_loss=expected_loss,actual_loss=actual_loss,model_exact=model_exact,optimizer_exact=equal(expected_opt,optimizer.state_dict()),cuda_rng_exact=equal(expected_rng,torch.cuda.get_rng_state_all()),cpu_rng_exact=torch.equal(expected_cpu_rng,torch.get_rng_state()),tensor_differences=differences,diagnostic_updates=3,production_updates=0)
        io.write(output/'replay_diagnostic.json',report)
        io.copy_verified(output/'replay_diagnostic.json',a.durable/'replay_diagnostic.json')
        print(json.dumps(report,indent=2),flush=True)
        return
"""+s[end:]
namespace=dict(__name__='diagnostic_worker',__file__=str(root/'worker.py'),a=a)
exec(compile(s,str(root/'worker.py'),'exec'),namespace)
namespace['configure'](protocol,output)
namespace['preflight'](protocol,folder,output)
