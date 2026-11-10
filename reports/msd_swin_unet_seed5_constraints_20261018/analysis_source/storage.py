"""Two-phase off-BeeGFS archive and narrowly scoped artifact retirement."""
import csv
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import time

from scripts.cross_dataset_campaign.common import read, write, digest


def quota():
    raw=subprocess.check_output(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'],text=True)
    found=re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB',raw)
    if not found:
        raise RuntimeError('Cannot verify personal BeeGFS quota')
    used,limit=map(float,found.groups())
    return dict(used_gib=used,limit_gib=limit,free_gib=limit-used,raw=raw,checked_at=time.time())


def verify_results(output, task):
    done=read(output/'completion.json')
    if done['status']!='complete':
        raise ValueError('Incomplete experiment cannot be archived/retired')
    if task['kind']=='conventional':
        config=read(output/'config.json')
        if config['augmentation']!='none' or done['consumed_training_flops']>config['budget_flops']:
            raise ValueError('Unrequested augmentation or cap exceeded')
        if done['discarded_attempts']:
            raise ValueError('Interrupted replay lost updates; retain recovery for inspection')
        for name,key in [('checkpoint_best.pt','checkpoint_best_sha256'),('checkpoint_latest.pt','checkpoint_latest_sha256')]:
            if digest(output/name)!=done[key]:
                raise ValueError('Checkpoint changed before archive')
        results={name:read(output/f'validation_{name}.json') for name in ('best','latest')}
        training=read(output/'training_final.json')
        if {r['case_name'] for r in training['cases']}!=set(config['train_cases']):
            raise ValueError('Training evaluation incomplete')
        for report in results.values():
            if {r['case_name'] for r in report['cases']}!=set(config['validation_cases']):
                raise ValueError('Validation cohort differs')
            if any('confusion' not in r or 'foreground_fn' not in r for r in report['cases']):
                raise ValueError('Required error diagnostics missing')
        write(output/'evaluation_diagnostics.json',dict(status='complete',evaluations=results,
            selected_step=done['selected_step'],stopping_step=done['updates'],
            primary_checkpoint='latest',ap_swaps_applicable=task['dataset']=='MSD'))
    else:
        protocol=read(output/'protocol.json')
        evaluation=read(output/'evaluation.json')
        expected={(split,c) for split,cases in [('training_support',protocol['train_cases']),('validation',protocol['evaluation_cases'])] for c in cases}
        if {(r['split'],r['case_name']) for r in evaluation['records']}!=expected:
            raise ValueError('Incomplete foundation evaluation')
        if digest(output/'checkpoint_latest.pt')!=done['checkpoint_sha256']:
            raise ValueError('Fine-tuned checkpoint changed')
        training=read(output/'training_summary.json')
        if training['updates']!=protocol['updates'] or done['updates']!=protocol['updates'] or training['frozen_before']!=training['frozen_after']:
            raise ValueError('Invalid fine-tuning audit')
        if len(read(output/'training_history.json'))!=protocol['updates']:
            raise ValueError('Training curve incomplete')


def prepare(output, task):
    path=output/'archive_manifest.json'
    if path.exists():
        return read(path)
    verify_results(output,task)
    metadata,bulky={},[]
    for item in sorted(output.rglob('*')):
        if item.is_dir():
            continue
        if item.is_symlink():
            raise ValueError('No symlink is a deletion candidate')
        relative=str(item.relative_to(output))
        info=dict(bytes=item.stat().st_size,sha256=digest(item))
        if item.suffix in ('.pt','.tmp') or item.name.endswith('.nii.gz'):
            bulky.append(dict(name=relative,**info))
        elif item.suffix in ('.json','.csv','.jsonl','.log','.lock','.png'):
            metadata[relative]=info
        else:
            raise ValueError('Unclassified output: '+str(item))
    result=dict(status='ready',task=task,metadata=metadata,deletion_candidates=bulky,
                created_at=time.time(),completion_sha256=digest(output/'completion.json'))
    write(path,result)
    return result


def verify_archive(destination, manifest):
    for name,item in manifest['metadata'].items():
        path=destination/name
        if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
            raise ValueError('Archive differs: '+name)
        if path.suffix=='.json':
            read(path)
        elif path.suffix=='.jsonl':
            for line in path.read_text().splitlines(): json.loads(line)
        elif path.suffix=='.csv':
            with path.open(newline='') as stream: list(csv.DictReader(stream))
        elif path.suffix not in ('.png',):
            path.read_text()


def retire(output):
    manifest=read(output/'archive_manifest.json');ack=read(output/'archive_acknowledgment.json')
    if ack['status']!='verified' or ack['manifest_sha256']!=digest(output/'archive_manifest.json'):
        raise ValueError('No matching durable archive acknowledgment')
    if not ack['source_verified'] or ack['durable_archive'].startswith(('/mnt/beegfs','/home/3160552','/tmp')):
        raise ValueError('Archive must be durable and outside BeeGFS')
    verify_archive(output,manifest)
    journal=output/'cleanup.json'
    previous=read(journal) if journal.exists() else None
    for item in manifest['deletion_candidates']:
        path=output/item['name']
        if path.parent.resolve()!=output.resolve() or path.is_symlink():
            raise ValueError('Deletion scope violation')
        if path.exists():
            if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
                raise ValueError('Artifact changed since archive')
        elif previous is None:
            raise ValueError('Artifact missing without a cleanup journal')
    receipt=dict(status='in_progress',deleted=manifest['deletion_candidates'],durable_archive=ack['durable_archive'],
                 released_file_bytes=sum(r['bytes'] for r in manifest['deletion_candidates']),time=time.time())
    write(journal,receipt)
    for item in manifest['deletion_candidates']:
        (output/item['name']).unlink(missing_ok=True)
    receipt['status']='complete';write(journal,receipt)
    return receipt
