"""Unchanged scientific worker plus read-only hashes at every native update."""
import os
from pathlib import Path
from scripts.colab_hfh import nnunet_search13_shared_worker as shared
from scripts.colab_hfh.runtime import read,write


class EveryNativeAnchor(shared.NativeAnchors):
    def finish(self,image,label,device,amp):
        super().finish(image,label,device,amp)
        step=self.rows[-1]['update']
        if step not in (20,30,45,60):
            from scripts.cross_dataset_campaign.capped_core import model_digest
            write(self.output/f'native_update{step}.json',dict(after_update=step,model_state_sha256=model_digest(self.model)))


if __name__=='__main__':
    import sys
    import torch
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('HFH_SEARCH_GPU_FRACTION','.06')))
    shared.NativeAnchors=EveryNativeAnchor
    path=Path(sys.argv[sys.argv.index('--protocol-dir')+1])
    is_native=read(path/'protocol.json')['active_objective']=='dice'
    exec(compile(shared.source(is_native),__file__,'exec'),{'__name__':'__main__','__file__':__file__})
