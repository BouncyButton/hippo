# Evaluation

This folder contains scripts used after training.

| File | Purpose |
| --- | --- |
| `evaluate.py` | Main evaluator for trained methods. The pipeline scripts call this automatically. |
| `evaluate_unetrpp.py` | UNETR++-specific evaluation helper. |
| `create_nnunet_metadata.py` | Packages nnUNet metadata for W&B. |
| `view.py` | Small helper for viewing outputs. |

Most students should use the pipeline scripts instead of calling these files
directly:

```bash
scripts/run_nnunet_pipeline.sh --dataset Dataset102_MNI --mode sanity
scripts/run_unetrpp_pipeline.sh --dataset Dataset102_MNI --mode sanity
```

Evaluation outputs are written to the run folder, usually under
`../hippopotamus_runs/.../evaluation_output`.

