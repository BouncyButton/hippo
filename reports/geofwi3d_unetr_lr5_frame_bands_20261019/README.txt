UNETR LR5 frame-excluded BANDS

One matched development run, fold 0 seed 0, peak LR 5e-4.
Primary controls: archived baseline final 50.6198003165%; original BANDS final 50.6070459552%.
Only training change: BANDS excludes the outer two-voxel observation frame.
Recalibration uses the original Dice+CE gradient target .10, p95 cap .50.
407 full-budget updates, no halfway gate. Training+calibration charge 181767152273416 FLOPs against cap 182178126543160.
Three GPU diagnostic preflight steps are reported separately under the established convention.

frame_cpu_validation.json: native-size loss and gradient, support checks on all 19 training cases, differential FLOP audit.
bundle_validation.json: frozen imports, archived references, cap and schedule verification.
protocol_difference.json and frame_loss.patch: exact changes.
package.json: frozen ZIP and notebook checksums.
launch_record.json: account, Drive artifacts and launch status.
