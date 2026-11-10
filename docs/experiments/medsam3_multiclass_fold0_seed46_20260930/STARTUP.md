# Checkpoint delivery and first-attempt failure

Job 677087 stopped before training because the streamed base checkpoint had the wrong size. Its quota preflight had 1.27 GiB free against the 0.9 GiB requirement; quota was not the failure. The failed job status, preflight and logs are preserved on the cluster.

The successful earlier MedSAM3 jobs already staged the base checkpoint on the login node at `/tmp/medsam3-3160552-staging/sam3.cdn.pt`. This file still exists and was SHA256-verified on 30 September 2026:

`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`

Use Slurm `sbcast --jobid=JOB --force SOURCE DESTINATION` from that file to the new job's private `/dev/shm/medsam3-ap-*/sam3.pt`, taking the destination from its status JSON. Only after successful checkpoint broadcast, broadcast `/mnt/beegfsstudents/home/3160552/medsam3_runtime_20260928/sam3-ready.json` to the ready path. The worker independently checks source integrity, quota, checkpoint size and then checkpoint SHA256 before training.

The earlier fold-3 launch records show checkpoint-plus-ready broadcasts completing in approximately 45–55 seconds. Avoid retransferring the checkpoint from the Mac through `srun` stdin. The replacement run retains the same scientific configuration and source.
