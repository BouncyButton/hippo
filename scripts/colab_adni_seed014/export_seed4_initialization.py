"""Run on the original CPU environment; emit portable ADNI initialization evidence."""
import argparse
import base64
import contextlib
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    for path in reversed([args.stage/'source', args.stage/'source/scripts',
                          args.stage/'source/baselines/nnUNet']):
        sys.path.insert(0, str(path))
    import torch
    import monai
    from monai.networks.layers import weight_init
    from scripts.cross_dataset_campaign import conventional as exp
    from scripts.cross_dataset_campaign.common import MODELS
    torch.set_num_threads(2)

    def sha(tensor):
        return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    result = dict(version=1, dataset='ADNI', python=platform.python_version(),
                  torch=torch.__version__, monai=monai.__version__,
                  cpu=torch.backends.cpu.get_cpu_capability(), threads=torch.get_num_threads(), models={})
    original = weight_init._no_grad_trunc_normal_
    for name in MODELS:
        folder = args.stage/'protocols'/name
        protocol = json.loads((folder/'protocol.json').read_text())
        architecture = json.loads((folder/'nnunet_architecture.json').read_text())
        result['models'][name] = {}
        for seed in (4,):
            calls = []
            def capture(tensor, mean, std, a, b):
                before = sha(torch.get_rng_state())
                value = original(tensor, mean, std, a, b)
                calls.append(dict(shape=list(value.shape), dtype=str(value.dtype), mean=mean,
                    std=std, a=a, b=b, rng_before=before, rng_after=sha(torch.get_rng_state()),
                    data=base64.b64encode(value.detach().numpy().tobytes()).decode()))
                return value
            exp.base.seed_everything(seed)
            weight_init._no_grad_trunc_normal_ = capture
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    model = exp.build_model(protocol, architecture)
            finally:
                weight_init._no_grad_trunc_normal_ = original
            digest = hashlib.sha256()
            tensors = {}
            for key, value in sorted(model.state_dict().items()):
                digest.update(key.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
                tensors[key] = dict(shape=list(value.shape), dtype=str(value.dtype), sha256=sha(value))
            result['models'][name][str(seed)] = dict(model_sha256=digest.hexdigest(),
                rng_after_model=sha(torch.get_rng_state()), calls=calls, tensors=tensors)
            print(name, seed, digest.hexdigest(), file=sys.stderr, flush=True)
            del model
            gc.collect()
    json.dump(result, sys.stdout, allow_nan=False)


if __name__ == '__main__':
    main()
