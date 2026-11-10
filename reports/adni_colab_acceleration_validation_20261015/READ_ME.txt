ADNI GPU equivalence validation — use your EXISTING prepared Colab runtime.

1. Wait until the ADNI campaign stops cleanly and all recovery uploads finish.
   The validator refuses active controllers and never interrupts them.
2. Upload Run_ADNI_ACCELERATION_VALIDATION.py with Colab's Files panel.
3. In Colab Terminal run:
   /content/hippo_adni_venv/bin/python -u /content/Run_ADNI_ACCELERATION_VALIDATION.py
4. Download the ZIP shown after SEND THIS ZIP BACK IN CHAT and attach it here.
   On failure, send that ZIP too; it includes the log.

No package installation, dataset upload, campaign run, changed charge ledger,
new scientific preflight receipt, or changed results directory. The code uses
only the current bundle/venv and reads the current environment and preflight.
It performs at most 24 disposable diagnostic optimizer steps across four models
(original/candidate paths), original ten-crop calibration on each path, and
binary evaluation/checkpoint replay. These are validation costs, not free GPU
work, but no additional study experiments are added. It stops at the first
failure. No tolerances are loosened. GPU caching, SDF/depth caching, deferred
metric readback and unchanged-best-checkpoint suppression remain disabled for
campaign training until this report has been checked.

The all-slot scheduler/NVMe/archive changes have passed local tests. The complete
training notebook/ZIP will be finalized with only the acceleration that passes
these GPU checks. Original scientific/data ZIPs and results_noaug_v1 remain.
