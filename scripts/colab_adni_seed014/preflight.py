"""Run only required diagnostics, with immutable old/new environment bindings."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    for name in ('parent-bundle','parent-results','bundle','durable'):
        parser.add_argument('--'+name,required=True,type=Path)
    a=parser.parse_args();package=Path(__file__).resolve().parent
    sys.path.insert(0,str(a.parent_bundle))
    import runtime
    runtime.activate(a.parent_bundle)
    environment=runtime.environment()
    with ExitStack() as stack:
        for root,results in ((a.parent_bundle,a.parent_results),(a.bundle,a.durable)):
            lock=stack.enter_context((root/'work/sweep.lock').open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (results/'.adni_controller_lock').exists() or (results/'.adni_seeds014_controller_lock').exists():
                raise RuntimeError('Existing controller ownership; preserve and inspect before preflight')
            expected=dict(bundle_sha256=runtime.digest(root/'bundle_manifest.json'),environment=environment)
            marker=results/'environment_binding.json'
            if marker.exists() and runtime.read(marker)!=expected:
                raise RuntimeError('GPU/software/bundle changed. Existing binding is preserved.')
        print('PHASE: verify original bundle and seed-only derivation',flush=True)
        runtime.verify_bundle(a.parent_bundle)
        spec=importlib.util.spec_from_file_location('seed4_derivation',a.bundle/'derivation.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.verify(a.bundle,a.parent_bundle)
        for root,results in ((a.parent_bundle,a.parent_results),(a.bundle,a.durable)):
            print('PHASE: required preflight for',root,flush=True)
            subprocess.run([sys.executable,'-u',str(root/'runtime.py'),'preflight','--bundle',str(root),'--durable',str(results)],check=True)
        # Check original evidence without altering its preflight or binding.
        sys.path.insert(0,str(package/'control04'))
        gate=__import__('acceleration_gate')
        binding=dict(bundle_sha256=runtime.digest(a.bundle/'bundle_manifest.json'),environment=environment)
        receipt=gate.verify(package/'control04',a.bundle,binding,runtime.read(a.durable/'preflight.json'))
        runtime.write(a.durable/'seed4_derivation_receipt.json',receipt)
        print('PASSED: original preflight preserved; separate seed4 preflight and backend derivation verified.',flush=True)


if __name__=='__main__':main()
