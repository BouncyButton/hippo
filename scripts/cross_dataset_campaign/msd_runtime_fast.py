"""Exact-result acceleration for the fixed, unaugmented MSD campaign.

The measured optimization graph, optimizer, precision, batching, validation
cadence and checkpoint cadence are unchanged. Only immutable annotation/data
work, host synchronization and redundant writes are avoided.
"""
from contextlib import nullcontext
from functools import partial
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.cross_dataset_campaign.common import confusion_metrics


@torch.no_grad()
def evaluate(model, dataset, device, amp, classes, batch_size=1):
    """Same per-case GPU operations, with one final host readback of metrics."""
    model.eval()
    matrices, scores, finite, names = [], [], [], []
    # Retain DataLoader, batch size and iterator RNG consumption exactly.
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        with torch.autocast('cuda', dtype=torch.bfloat16) if amp else nullcontext():
            logits = model(batch['image'].to(device))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.shape[1] != len(classes):
            raise ValueError('Invalid validation output shape')
        finite.append(torch.isfinite(logits).all())
        for index in range(len(batch['case_name'])):
            probabilities = logits[index:index+1].float().softmax(1)
            prediction = probabilities.argmax(1)
            truth = batch['label'][index:index+1].to(device)[:, 0].long()
            matrices.append(torch.bincount((truth*len(classes)+prediction).flatten(),
                minlength=len(classes)**2).reshape(len(classes),len(classes)))
            case_scores = []
            for c in range(1,len(classes)):
                gt = truth == c
                case_scores.append(2*(probabilities[:,c]*gt).sum()/
                    (probabilities[:,c].sum()+gt.sum()).clamp_min(1e-12))
            scores.append(torch.stack(case_scores))
            names.append(batch['case_name'][index])
    if not names or not bool(torch.stack(finite).all()):
        raise ValueError('Empty cohort or invalid validation output')
    matrices = torch.stack(matrices).cpu().tolist()
    scores = torch.stack(scores).cpu().tolist()
    records = []
    for name,matrix,soft in zip(names,matrices,scores):
        row = confusion_metrics(matrix,classes)
        row['case_name'] = name
        for c,label in enumerate(classes[1:]):
            row[label+'_hard_dice'] = row['per_class'][label]['dice']
            row[label+'_soft_dice'] = soft[c]
        row['macro_soft_dice'] = float(np.mean(soft))
        records.append(row)
    keys = [k for k in records[0] if k.endswith('_dice')]
    return dict(summary={k:float(np.mean([r[k] for r in records])) for k in keys},cases=records)


class RuntimeAcceleration:
    """Scoped hooks; caches own tensor references and reject mutated labels."""
    def __init__(self,experiment,protocol,device):
        if protocol['dataset']!='MSD' or protocol['augmentation']!='none':
            raise ValueError('Caching is authorized only for fixed unaugmented MSD crops')
        self.experiment,self.protocol,self.device = experiment,protocol,torch.device(device)
        self.originals = {}
        self.caches = {'sdf':{},'depth':{}}
        self.best = {}
        self.stats = dict(cached_data_bytes=0,sdf_hits=0,sdf_misses=0,depth_hits=0,depth_misses=0,
            skipped_best_checkpoint_writes=0,checkpoint_write_seconds=0.,evaluation_seconds=0.)

    @staticmethod
    def key(labels):
        return (str(labels.device),labels.untyped_storage().data_ptr(),labels.storage_offset(),
            tuple(labels.shape),tuple(labels.stride()),str(labels.dtype),labels._version)

    def cached(self,name,function,labels,*args):
        extra = () if not args else (np.asarray(args[0]).tobytes(),json.dumps(args[1],sort_keys=True))
        key = (self.key(labels),extra)
        cache = self.caches[name]
        if key not in cache:
            self.stats[name+'_misses'] += 1
            value = function(labels,*args)
            # A strong label reference prevents storage-pointer reuse. Values
            # are immutable annotation fields, never predictions or gradients.
            cache[key] = (labels,value)
        else:self.stats[name+'_hits'] += 1
        return cache[key][1]

    def install(self):
        base,trainer = self.experiment.base,self.experiment.trainer
        for obj,name in ((base,'PreparedPatients'),(base,'evaluate'),(base,'save_checkpoint'),
                         (trainer,'signed_distance_maps'),(trainer,'geometry_weights')):
            self.originals[(obj,name)] = getattr(obj,name)
        old_dataset = base.PreparedPatients
        owner = self
        class CachedPatients(old_dataset):
            def __init__(self,dataframe,names):
                super().__init__(dataframe,names)
                for item in self.items:
                    for key in ('image','label'):
                        item[key] = item[key].to(owner.device)
                        owner.stats['cached_data_bytes'] += item[key].numel()*item[key].element_size()
        base.PreparedPatients = CachedPatients
        trainer.signed_distance_maps = partial(self.cached,'sdf',trainer.signed_distance_maps)
        trainer.geometry_weights = partial(self.cached,'depth',trainer.geometry_weights)
        def measured_evaluate(*args,**kwargs):
            start=time.perf_counter()
            try:return evaluate(*args,**kwargs,classes=self.protocol['classes'],
                batch_size=self.protocol.get('runtime_acceleration',{}).get('validation_batch_size',1))
            finally:self.stats['evaluation_seconds'] += time.perf_counter()-start
        base.evaluate = measured_evaluate
        original_save = base.save_checkpoint
        def save(path,value):
            signature = (path.stat().st_size,path.stat().st_mtime_ns) if path.exists() else None
            previous = self.best.get(str(path))
            if path.name=='checkpoint_best.pt' and previous and previous[0] is value and previous[1]==signature:
                self.stats['skipped_best_checkpoint_writes'] += 1
                return
            start=time.perf_counter()
            original_save(path,value)
            self.stats['checkpoint_write_seconds'] += time.perf_counter()-start
            if path.name=='checkpoint_best.pt':
                self.best[str(path)] = (value,(path.stat().st_size,path.stat().st_mtime_ns))
        base.save_checkpoint = save
        return self

    def close(self):
        for (obj,name),value in self.originals.items():setattr(obj,name,value)
        self.originals.clear()
        self.caches = {'sdf':{},'depth':{}}
        self.best.clear()

    def __enter__(self):return self.install()

    def __exit__(self,*_):self.close()
