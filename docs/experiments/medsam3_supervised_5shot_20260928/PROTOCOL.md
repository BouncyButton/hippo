# MedSAM3: supervised five-volume constraint ablation

The three arms start independently from the same released medical LoRA weights:
native SAM3 objective; native + bands; native + bands + edge. Only LoRA parameters
are trained. Constraints enter backpropagation; inference has no mask refinement.

The fixed support and validation cases, seed and optimizer settings are in
`protocol.json`. Each arm receives 200 AdamW updates, two slice pairs per update
(800 slice presentations), weight decay 0.01, gradient clipping at 1, and bf16.
Learning rate warms for 20 updates to 5e-5, then falls by cosine to 5e-6. The final
adapter is evaluated; validation never selects checkpoints or coefficients.

Training labels build the original two-step, six-connected inner and outer 3D
bands on each native canonical volume. Bands uses balanced inner/outer BCE,
without focal or degree weighting. Edge uses the original squared difference of
signed foreground residuals across positive-axis faces whose endpoints both lie
in the bands. Full-volume side counts and face counts are retained.

For each microstep, select a support volume uniformly, then an axial anchor
uniformly. Forward the anchor and its cyclic successor, and average their native
SAM3 objectives. Cyclic pairing gives uniform expected native slice exposure.
Bands uses the anchor plane; edge uses in-plane faces and the next-plane faces,
excluding any anatomical wraparound. Multiply these contributions by the volume
depth. Their expectations equal the original full-volume losses; tests compare
both loss values and gradients on non-cubic, empty, full and border-touching data.

SAM3 predicts instances. A differentiable semantic foreground field is the max
of native-size sigmoid mask logits multiplied by query and presence probabilities.
This representation adaptation is necessary for the original semantic losses.
No threshold, ground-truth matching, box prompt or point prompt enters this field.
BCE is computed in log-probability space, preserving corrective gradients even
at very low confidence. Edge uses the corresponding unclipped probabilities.
Native SAM3 losses retain upstream matching and their existing weights.

Coefficients are calibrated once at initialization, using four deterministic
band-bearing anchors from each of the five support volumes. Set bands' median
LoRA gradient norm ratio to 0.1 relative to native loss, capped at a 0.5 ratio at
the 95th percentile. Calibrate edge the same way relative to native + weighted
bands, including their gradient dot product. Zero constraint-gradient samples
are omitted and recorded; at least five nonzero samples are required. No model
updates occur during calibration. Both constraint coefficients ramp over 20
updates. Bands has the identical coefficient in both constrained arms.

Reset adapter weights and all RNGs before each arm; replay the exact same saved
volume/anchor schedule and use separate fresh optimizers. Record parameter hashes,
nonzero calibration gradients, per-update component losses and coefficients,
exposures, trained adapters, probabilities, masks and Dice confusion counts.

Inference uses only images and the text prompt `hippocampus`, with the same 0.5
confidence/NMS/mask settings across arms. Save all predictions for all arms before
opening validation labels. Score whole-hippocampus union Dice on five existing
development-validation volumes. This is a one-seed pilot, not a new independent
test cohort or a statistically conclusive estimate. These are five labelled
volumes, not five slices; patient identities are unavailable.

One Slurm GPU allocation runs calibration and all three arms sequentially.
Require at least 2 GiB free BeeGFS quota before submission and at worker startup.
Keep the base checkpoint and temporary caches on node-local storage; retain only
one final LoRA adapter per arm on BeeGFS.
