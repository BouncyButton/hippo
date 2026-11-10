import contextlib, gc, hashlib, importlib.metadata, io, json, platform
from pathlib import Path
import torch
from scripts.cross_dataset_campaign import conventional as exp
from scripts.cross_dataset_campaign.common import MODELS, read
parent=Path('/mnt/beegfsstudents/home/3160552/adni_msd_four_models_seed3_baselines_20261015')
torch.set_num_threads(2)
records=[]
for seed in (7,8,9):
 for model_name in MODELS:
  folder=parent/'protocols'/f'MSD_{model_name}_f0s3'
  p=read(folder/'protocol.json')
  exp.base.seed_everything(seed)
  with contextlib.redirect_stdout(io.StringIO()):
   model=exp.build_model(p,read(folder/'nnunet_architecture.json'))
  h=hashlib.sha256()
  for name,value in sorted(model.state_dict().items()):
   h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
  records.append(dict(dataset='MSD',model=model_name,seed=seed,model_sha256=h.hexdigest(),parameters=sum(v.numel() for v in model.parameters())))
  del model
  gc.collect()
print(json.dumps(dict(torch=str(torch.__version__),monai=importlib.metadata.version('monai'),python=platform.python_version(),optimizer_updates=0,models=records),indent=2))
