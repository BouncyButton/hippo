# Thesis audit — hippocampus segmentation with loss constraints

**Date:** 2026-09-06 · **Scope:** every training run and every research direction in
`~/Desktop/hippo`, from the Family-B baselines through the currently-running translation
equivariance replication.

Everything below is traced to a file in the repository. Where a number lives only on the
cluster and could not be re-read from here, it is marked **[cluster-only]**.

---

## 0. Read this first — the three rules that govern every comparison

Three traps have already cost this project one retracted headline. They are recorded in
`COMPARABILITY_REGISTER_20260903.md` and `PROJECT_PROMPT_20260905.md`, and they constrain
every table in this document.

**(a) Configuration families.** Two incompatible configurations exist and they differ by
**+0.0041** hard macro Dice — larger than any constraint effect ever measured here.

| | batch size | AMP | source hash | contents |
|---|---:|---|---|---|
| **Family A** | 2 | off | not recorded | 2 × `none` controls, 2 × translation equivariance |
| **Family B** | 1 | on | recorded | everything else, including the official baseline |

Decision of 2026-09-03: **Family B is the official line.** Never compare absolute Dice across
families; compare effect-against-own-control only.

**(b) Exposure.** 5-epoch, 30-epoch and 50-epoch runs are different experiments. The warm-up
is `α_e = min(1, e/5)`, so a 5-epoch pilot observes exactly **one** epoch at full constraint
strength. Measured seed noise at epoch 5 is **0.0129** — larger than every constraint effect
ever measured at that exposure. Every focal gate and the one-cut verdict was decided there.

**(c) Precision and pooling.** fp16 (cluster/AMP) versus fp32 (local CPU) moves macro Dice by
**0.0067** on the *identical* checkpoint. Per-case-averaged versus voxel-pooled differ by
~0.006. Cluster per-case baseline reads **0.874763**; local voxel-pooled fp32 reads
**0.868649**. Both correct, different pools. Quote deltas within one precision and one pooling.

**Noise floor** (first measured 2026-09-04, jobs 648443/648444):

| window | mean absolute seed gap | max |
|---|---:|---:|
| epochs 1–5 | 0.048354 | 0.135400 |
| epochs 16–25 | 0.002161 | 0.007090 |
| epochs 26–35 | 0.000936 | 0.001497 |
| **epochs 41–50** | **0.001207** | 0.001901 |

**Nothing below ~2 × 0.0012 at converged exposure can be claimed.**

---

## 1. Family B baselines

### 1.1 Fixed conditions, shared by every Family-B run

| item | value |
|---|---|
| dataset | MSD Task04 Hippocampus, `Dataset101_MSD`, 1 mm isotropic |
| classes | 0 background, 1 anterior (head), 2 posterior (body/tail) |
| pickle | `msd_hippocampus_full.pkl`, `sha256 3311d2233b…` |
| splits | `splits_final.json`, `sha256 1d6a3fe993…`, 5 folds |
| fold | **0 only** — 208 train / 52 val |
| model | MONAI SwinUNETR, 1 input channel, 3 output classes |
| input | 64³ crop, no resize (symmetric pad + centre crop) |
| batch size | **1** |
| AMP | **on** |
| optimizer | AdamW, lr 1e-4, wd 1e-5 |
| scheduler | StepLR, step 20, γ 0.5 |
| epochs | 50 (30 and 5 for the shorter arms) |
| supervised loss | `DiceLoss(to_onehot_y=True, softmax=True)` **and nothing else — there is no cross-entropy term anywhere** |
| objective | `total = dice + min(1, e/5) · λ · constraint` |
| augmentation | **none.** Pipeline is `EnsureChannelFirstd → NormalizeIntensity → DivisiblePadd → SpatialPadd → CenterSpatialCropd`. No `Rand*` transforms of any kind |
| GPU | A100 MIG 4g.40gb, partition `stud`, 24 h 10 m wall limit |

**What a seed actually varies.** `set_determinism(seed=0)` is a module-level constant in
`baselines/swin_unetr/swin_unetr.py`, independent of `--seed`. So `--seed` varies exactly two
things: network weight initialisation, and training sample order (`shuffle=True,
generator=train_generator`). It does **not** vary crops or augmentation, because neither is
random. Consequence: **training is deterministic given the seed**, and same-seed reruns
measure nothing.

### 1.2 The official baseline

**Run:** `matched_control_20260902/msd_fold0_none_matched_50epoch_seed0`
**Slurm:** 646983 · **Completed:** 2026-09-03, `COMPLETED`, 18:25:37 wall, 50/50 epochs
**Seed:** 0 · **Source:** `cf7d77bc714d99d96d9975eb84bfb0d38644cc6b7305e3defb017371953628bf`

| landmark | val Dice (hard macro) |
|---|---:|
| epoch 5 | 0.8461556 |
| **best (epoch 16)** | **0.8785989** |
| epoch 30 | 0.8760412 |
| epoch 50 | 0.8747625 |
| mean epochs 21–30 | 0.8769640 |
| mean epochs 41–50 | 0.8755147 |

**One run covers every exposure.** Because training is deterministic given the seed, a
*k*-epoch Dice-only control is exactly this run truncated at epoch *k*. This was verified, not
assumed: the two independent 5-epoch Dice-only controls —
`msd_fold0_none_calibration_pilot_seed0_20260806_220206` (source `347562ba…`) and
`onecut_protocol_20260902/msd_fold0_none_calibration_seed0` (job 646126, source `cf7d77bc…`) —
are **bit-identical to 16 significant digits across all five epochs**, and identical again to
this run's first five epochs. Both read 0.8461556 at epoch 5. That single fact retroactively
supplies a matched control for every bs=1 / AMP-on experiment in the project, and it also
closes the source-drift worry: two different code eras produced the same trajectory.

### 1.3 The complete Family-B baseline (Dice-only) inventory

| run | preset | epochs | seed | job | status |
|---|---|---:|---:|---|---|
| `matched_control_20260902/msd_fold0_none_matched_50epoch_seed0` | none | 50 | 0 | 646983 | official baseline |
| `seed1_replicate_20260903/msd_fold0_none_seed1` | none | 50 | 1 | 648443 | COMPLETED 2026-09-04 13:17:40, 17:50:41 wall |
| `msd_fold0_none_calibration_pilot_seed0_20260806_220206` | none | 5 | 0 | — | calibration control, source `347562ba…` |
| `onecut_protocol_20260902/msd_fold0_none_calibration_seed0` | none | 5 | 0 | 646126 | calibration control, source `cf7d77bc…` |

The two 5-epoch controls are **bit-identical prefixes** of the 50-epoch run — they are not
independent data and must never be counted as replicates.

The seed-1 control (648443) is the only genuine replicate in the whole project, and it is what
produced the noise floor in §0. **[cluster-only]** its per-epoch values are on the cluster; the
derived seed gaps are reproduced in §0 from `PROJECT_PROMPT_20260905.md`.

### 1.4 For contrast — the Family-A controls (not baselines for anything here)

`msd_fold0_none_20260730_124318_611097` (best 0.874906) and
`msd_fold0_none_20260806_110334_616958` (best 0.874524). Batch size 2, AMP off, 50 epochs,
seed 0, no source hash recorded. The 0.000382 gap between them is a **code-era difference, not
noise** — same seed, deterministic training. These serve the Family-A equivariance runs and
nothing else.

### 1.5 The binding limitation

Every Family-B result except the two seed-1 replicates is **fold 0, seed 0**. There is no
cross-fold evidence anywhere in the project, and fold-0 validation has driven essentially every
design decision — so it is development data, not a test set.

---

## 2. Research direction 1 — the outer-boundary band constraint

### 2.1 Why this was the first attempt

The fold-0 error analysis (`evaluation/fold0_voxel_error_results/`) on the 52 held-out cases
found 39,781 wrong voxels (765.0 per patient), decomposed as:

| error group | voxels | share |
|---|---:|---:|
| foreground false positives | 19,512 | 49.0% |
| foreground false negatives | 16,397 | 41.2% |
| anterior/posterior swaps | 3,872 | 9.7% |
| **all foreground/background errors** | **35,909** | **90.3%** |

and, decisively, **91.4% of all errors lie within one voxel of the GT outer contour and 98.4%
within two.** The dominant failure was not a distant false component or a class swap — it was
the foreground/background transition displaced by one or two voxels.

Under the exact two-step 6-connected masks the loss actually uses, **35,513 of 35,909
foreground errors (98.90%) fall on the correspondingly supervised side** (19,176/19,512 FP in
the outer band, 16,337/16,397 FN in the inner band). That is the direct empirical justification
for the guard domains.

**The plain-language rationale:** the model already knows *roughly* where the hippocampus is;
it gets the last voxel or two of the rim wrong. So supervise exactly that rim, from both sides,
with equal weight — push the two layers just inside the contour toward foreground and the two
layers just outside toward background — and leave everything else to Dice.

### 2.2 The constraint, stated as LogLTN

The canonical write-up is `docs/thesis/new_constraints/bands/Focal Log-LTN.md`; the implementation
is `bands/outer_boundary.py`.

**Language.** One unary predicate

$$H(v):\ \text{“voxel } v \text{ is hippocampal foreground.”}$$

The individual is formally $x_{b,v}=(X_b, v)$ — coordinate $v$ *in the context of image*
$X_b$ — because the prediction at $v$ depends on the whole patient volume. Patient context is
kept implicit as $H(v)$.

**Grounding of the predicate.** The three-class logits are $\mathbf z_b(v)=(z_0,z_A,z_P)$.
Anterior and posterior are disjoint outcomes of one softmax and together *are* the hippocampus,
so the union has model probability $p_A+p_P = 1-p_0$ — an exact marginalisation, not an
independence assumption. Its log-odds are

$$
r_{H,b}(v)=\operatorname{LSE}\bigl(z_{A,b}(v),z_{P,b}(v)\bigr)-z_{0,b}(v),
\qquad
\mathcal G_\theta(H(v)) = h_b(v)=\sigma(r_{H,b}(v)) = p_A+p_P .
$$

Proof that the sigmoid recovers exactly the summed foreground probability: with
$S_F=\sum_{c\in F}e^{z_c}$, $S_C=e^{z_0}$, $r_H=\log(S_F/S_C)$ and
$\sigma(r_H)=\frac{1}{1+S_C/S_F}=\frac{S_F}{S_F+S_C}=\sum_{c\in F}p_c$. The code evaluates
`logsumexp` directly and never divides by a small probability. Negation is standard:
$\mathcal G(\neg H(v))=1-h_b(v)=\sigma(-r_H)$.

**Guard domains (crisp, GT-derived, no gradient).** With the 7-position 6-connected cross
$C_6(v)$, one dilation $D_1$ and one erosion $E_1$, applied twice:

$$
B_{\mathrm{in},b}=F_b\setminus E_2(F_b),
\qquad
B_{\mathrm{out},b}=D_2(F_b)\setminus F_b,
$$

where $F_b=\{v: g_b(v)\in\{A,P\}\}$ is the GT foreground union. They are disjoint by
construction; $B_{\mathrm{in}}\subseteq F$ so every inner voxel is GT foreground, and
$B_{\mathrm{out}}\cap F=\varnothing$ so every outer voxel is GT background. They are built
under `no_grad` — the network must not learn to move its own supervision region. For a fixed
mask, $\partial_z[m(v)\ell(v)] = m(v)\,\partial_z\ell(v)$, so selected voxels backpropagate
normally and unselected ones get no band gradient.

**Knowledge base.** Two guarded universal formulas per patient:

$$
\phi_{\mathrm{in},b}=\forall_{v\in B_{\mathrm{in},b}}H(v),
\qquad
\phi_{\mathrm{out},b}=\forall_{v\in B_{\mathrm{out},b}}\neg H(v).
$$

These are **restricted quantifications over crisp domains**, not fuzzy implications
$B_{\mathrm{in}}(v)\rightarrow H(v)$ evaluated everywhere. Multiplying by the mask *is* the
tensor implementation of selecting the quantified instances; writing it as an implication would
force a choice of fuzzy implication and change the loss.

**Quantifier grounding (γ = 0, the run that was actually trained).** The normalised-product
universal aggregator

$$
S_0=\Bigl(\prod_{i=1}^{N}t_i\Bigr)^{1/N},
\qquad
L_0=-\log S_0=-\frac1N\sum_i\log t_i
$$

gives, on the two sides,

$$
L^{(0)}_{\mathrm{in},b}=-\frac1{n_{\mathrm{in}}}\!\!\sum_{v\in B_{\mathrm{in},b}}\!\!\log h_b(v),
\qquad
L^{(0)}_{\mathrm{out},b}=-\frac1{n_{\mathrm{out}}}\!\!\sum_{v\in B_{\mathrm{out},b}}\!\!\log\bigl(1-h_b(v)\bigr).
$$

**These are exactly binary cross-entropy with target 1 inside and 0 outside.** At γ = 0 the
implemented loss is simultaneously BCE and standard mean-logLTN negative log-satisfaction.

**Reductions.** Equal weight to the two formulas, then equal weight to valid patients:

$$
L_{\mathrm{case},b}=\tfrac12 L_{\mathrm{in},b}+\tfrac12 L_{\mathrm{out},b},
\qquad
L_{\mathrm{band}}=\frac{1}{|\mathcal V|}\sum_{b\in\mathcal V}L_{\mathrm{case},b}.
$$

Because $L=-\log S$, equal averaging in log space is *geometric* aggregation of satisfactions:
$S_{\mathrm{case},b}=\sqrt{S_{\mathrm{in},b}S_{\mathrm{out},b}}$ and
$S_{\mathrm{batch}}=\bigl(\prod_b S_{\mathrm{case},b}\bigr)^{1/|\mathcal V|}$. The code never
forms these products; it minimises the mean negative log-satisfaction. The two-stage average
matters: the outer band usually has more voxels than the inner one, and a single mean over
their union would let contour *expansion* dominate contour *contraction*.

A patient is valid only when both bands are non-empty
($v_n = \mathbf 1[n_{\mathrm{in}}>0]\,\mathbf 1[n_{\mathrm{out}}>0]$); invalid patients are
**skipped**, not scored zero, because a zero would silently dilute the batch mean. An all-invalid batch returns a graph-connected
float32 zero so backward still runs.

**Full objective.**

$$
L_{\mathrm{total}} \;=\; L_{\mathrm{Dice}} \;+\; s(e)\,\lambda_{\mathrm{band}}\,L_{\mathrm{band}},
\qquad
s(e)=\min\!\Bigl(1,\;\tfrac{e}{5}\Bigr).
$$

**Why log space.** The implementation minimises $-\log S$, not $1-S$. Differentiating
$1-\exp(-L)$ multiplies the gradient by the global satisfaction $\exp(-L)$, which becomes tiny
exactly when the formula is badly violated. The reported `truth = exp(-case_loss)` is computed
**after** the loss, for diagnostics only. *(This is the single most important design point to
carry into §6: the equivariance constraint does the opposite.)*

### 2.3 How the variables were chosen

| variable | value | how it was chosen |
|---|---|---|
| foreground grouping $F=\{A,P\}$ | union | deliberate — the constraint addresses foreground/background placement and delegates A/P to Dice. The loss is invariant to swapping the A and P channels |
| band steps | **exactly 2** | from the error histogram: 98.4% of errors within Euclidean distance 2; 98.90% of FP/FN fall inside the exact 2-step masks. The public loss *requires* `steps=2` |
| connectivity | 6-connected cross, applied twice (Manhattan r = 2) | matched morphological comparison on the 52 saved error maps: Manhattan covers 35,513/35,909 (98.90%) with 300,275 band voxels; Euclidean r=2 covers 35,591 (99.11%) with 315,978; Chebyshev covers 35,861 (99.87%) with **520,811** (+73.4% voxels for 348 more errors). Manhattan is the conservative choice |
| side weights | ½ / ½ | equal nominal weight to expansion and contraction |
| patient weights | equal | so large hippocampi do not dominate |
| $\varepsilon$ | 1e-6 | numerical only |
| **λ_band** | **0.011506333633759837** | **not** tuned on validation Dice — see below |

**The λ calibration ritual** (`bands/calibrate_weight.py`). On at most 32 deterministic
*training* cases, from a frozen epoch-5 Dice-only checkpoint with matching data, fold,
transforms, AMP policy, runtime and source, measure gradients **with respect to the logits**
(not the parameters):

$$
G^{(n)}_D=\sqrt{\tfrac1M\textstyle\sum_j(\partial L_{\mathrm{Dice}}^{(n)}/\partial z_j)^2},
\qquad
G^{(n)}_B=\sqrt{\tfrac1M\textstyle\sum_j(\partial L_{\mathrm{band}}^{(n)}/\partial z_j)^2},
$$

then with $D_{50}=\mathrm{median}_n G_D$, $B_{50}=\mathrm{median}_n G_B$,
$B_{95}=Q_{0.95}(G_B)$:

$$
\lambda_{\mathrm{band}}=\min\underbrace{\Bigl(0.10\,\tfrac{D_{50}}{B_{50}}\Bigr)}_{\text{target: 10\% of Dice gradient}}\;,\;
\underbrace{\Bigl(0.50\,\tfrac{D_{50}}{B_{95}}\Bigr)}_{\text{cap: p95 band}\le 50\%\text{ of median Dice}} .
$$

The trainer refuses to start unless the calibration report's source hash, checkpoint,
config, dataset/split hashes, runtime, class grouping and focal γ all match, and the numeric
weight matches the report exactly. Recalibration under the newer source on 2026-09-03 (job
648436) reproduced **λ = 0.011506333633759837 bit-identically**, independently confirming the
γ = 0 path is numerically unchanged across code eras.

**Known weakness:** λ is fixed once and held for all 50 epochs, while telemetry shows the
useful auxiliary gradient is spent by roughly epoch 25.

### 2.4 The bands runs and their results

| run | epochs | seed | λ | final | best | ep21–30 | ep41–50 |
|---|---:|---:|---|---:|---:|---:|---:|
| `msd_fold0_bands_pilot_seed0_20260807_082555` | 5 | 0 | 0.011506333633759837 | 0.8538741 | 0.8538741 | — | — |
| `msd_fold0_bands_final_seed0_20260807_104915` | 50 | 0 | 0.011506333633759837 | 0.8762991 | 0.8803044 (ep 22) | 0.8788906 | 0.8766739 |
| `seed1_replicate_20260903/msd_fold0_bands_seed1` (648444) | 50 | 1 | 0.011506333633759837 | **[cluster-only]** | best at ep 15 | — | — |

Against the official baseline at matched exposure:

| exposure | Δ final | Δ best | Δ ep21–30 | Δ ep41–50 |
|---|---:|---:|---:|---:|
| 5 epochs | **+0.0077184** | +0.0077184 | — | — |
| 50 epochs, seed 0 | **+0.0015366** | +0.0017055 | +0.0019266 | +0.0011592 |

**The two-seed verdict (`PREREG_BANDS_REPLICATION_20260902.md` §8c, resolved 2026-09-05):**

| endpoint | Δ seed 0 | Δ seed 1 | sign | two-seed mean |
|---|---:|---:|---|---:|
| **epoch 50 (pre-registered primary)** | +0.001537 | **−0.001248** | **FLIP** | **+0.000144** |
| ep21–30 mean | +0.001927 | −0.000082 | FLIP | +0.000922 |
| ep41–50 mean | +0.001159 | −0.000880 | FLIP | +0.000140 |
| best epoch (secondary) | +0.001705 | +0.000830 | same | +0.001268 |

**The sign flips on three of four endpoints including the primary. This is a failure to
replicate.** The originally reported **+0.0058** decomposes as ~72% configuration (batch size
and AMP, worth +0.0041) plus a +0.0015 remainder that does not survive a change of seed. The
only surviving endpoint is best-epoch, which carries max-selection bias on a validation fold
already used for model selection.

The constraint also **destabilises training** — seed-to-seed gap within arm:

| endpoint | `none` | `bands` | ratio |
|---|---:|---:|---:|
| epoch 50 | 0.000787 | 0.003572 | 4.5× |
| ep21–30 | 0.000530 | 0.002539 | 4.8× |
| ep41–50 | 0.001207 | 0.003246 | 2.7× |

Peak epoch wanders 22 → 15 for bands, where the control moves only 16 → 21.

**The geometry panel at matched epoch 50** (job 648417, exact common evaluator) is the
most informative single table in the boundary line. **Both arms are seed 0** — the baseline is
`matched_control_20260902/msd_fold0_none_matched_50epoch_seed0` (646983) and the bands arm is
`msd_fold0_bands_final_seed0_20260807_104915`:

| metric | baseline, **seed 0** | bands, **seed 0** | Δ |
|---|---:|---:|---:|
| union Dice | 0.896747 | 0.896745 | **−0.000002** |
| anterior / posterior Dice | 0.881484 / 0.868048 | 0.883206 / 0.869381 | +0.0017 / +0.0013 |
| surface Dice 1 mm | 0.956845 | 0.956781 | −0.000064 |
| ASSD (mm) | 0.434128 | 0.439121 | +0.004993 (worse) |
| FP / FN | 16,517 / 18,937 | 17,713 / 17,953 | +1,196 / −984 |
| **A/P swaps** | 3,788 | 3,518 | **−270** |
| **connected components** | 67 | 102 | **+35 (+52%)** |
| total decoded errors | 39,242 | 39,184 | −58 (0.15%) |

**No equivalent panel exists for seed 1** — it was never computed. So this table describes the
one seed whose hard-Dice delta was *positive* (+0.001537); on seed 1 that delta is −0.001248,
and nothing is known about how these geometry metrics moved there. Read the panel as a
mechanistic description of the seed-0 run, not as a replicated property of the constraint. Note
also that the two arms differ in source era (bands `347562ba…`, baseline `cf7d77bc…`), a
limitation recorded in the pre-registration's amendment log.

**The outer boundary — the thing the constraint explicitly supervises — is untouched. The
entire hard-Dice gain is A/P reassignment, and components regress 52%.** This inverts the
standing interpretation, which had called A/P swaps the band's blind spot.

### 2.5 Three formulation findings that explain the null

1. **The band is not boundary-local.** Measured on the 52 GT labels at `steps=2`: the inner
   band is **2,364 of 3,353 voxels — 70.5% of the whole hippocampus** — and the outer band
   (3,410 voxels) is *larger than the structure itself*. Together 1.7× its volume.
2. **The loss is exactly cross-entropy**, added to a pipeline whose only supervised term is
   `DiceLoss`. The accurate description is *class-balanced grouped cross-entropy with a
   geometrically-defined hard-negative population* — i.e. it is close to "add CE to Dice", the
   standard nnU-Net recipe, with a mild spatial reweighting. **The control that separates those
   two readings has never been run.**
3. **The active ingredient is the denominator.** A frozen-logit audit found **99.94%** of the
   *unmasked* objective's gradient already lies inside the band, yet cos(banded, unmasked)
   = 0.845 and banded repairs better at every step size. The mechanism is that the band averages
   negatives over ~3,400 near-contour voxels instead of ~258,000 — a **76× concentration of
   anti-FP pressure**. This is the direct explanation for §3: the band already performs
   hard-negative selection, so focal weighting is redundant.

   *In plain terms.* Only uncertain voxels produce gradient at all — deep interior has
   $p_H\approx 1$ with target 1, distant background has $p_H\approx 0$ with target 0, and both
   give $\sigma(r_H)-y\approx 0$. Those uncertain voxels all sit near the contour, which is why
   masking to the band removes almost no gradient mass. What masking *does* change is what the
   two side-means divide by. Recomputed over all 260 labels at `steps=2`:

   | side | unmasked denominator | banded denominator | shrink |
   |---|---:|---:|---:|
   | inner (foreground) | $\lvert F\rvert$ ≈ 3,295 | $\lvert B_{\mathrm{in}}\rvert$ ≈ 2,315 | 1.42× |
   | outer (background) | $64^3-\lvert F\rvert$ ≈ 258,849 | $\lvert B_{\mathrm{out}}\rvert$ ≈ 3,347 | **78.1×** |

   The inner half is essentially unchanged; the outer half is amplified ~78×, i.e. **~55×
   relative to the inner half**. Unmasked, a genuinely wrong near-contour background voxel is
   averaged against a quarter-million already-correct ones and is diluted to nothing; banded,
   the same error carries ~78× the weight. The band does not change *which* voxels are pushed,
   only *how hard* — it is a hard-negative miner wearing a boundary prior's clothes.

   *And that is precisely why focal added nothing.* Focal is also a hard-negative miner: it
   down-weights easy voxels in the numerator while leaving them in the count $N$. The band had
   already deleted them from numerator **and** denominator, so by the time focal is applied the
   ~255,000 easy background voxels are gone and the surviving population is uniformly
   uncertain. The two criteria also select the same voxels — the band by geometry (within two
   steps of the contour), focal by difficulty (currently wrong) — and §2.1 measured those to
   coincide, with 91.4% of all errors within one voxel of the contour. Focal's selection is
   therefore nested inside the band's and contributes only variance: telemetry shows it
   concentrating 94–99% of the auxiliary gradient into the hardest 10% of band voxels, which
   is what makes its validation curve 2–3× noisier and its best-epoch maximum look like a win.
   **The band and focal are the same intervention applied twice.**

There is also a mechanical reason the grouped log-odds cannot fix swaps, and may entrench them.
Since $\partial r_H/\partial z_A = p_A/(p_A+p_P)$ and
$\partial r_H/\partial z_P = p_P/(p_A+p_P)$, the "be foreground" gradient is distributed
between anterior and posterior **in proportion to the model's current belief** — a boundary voxel wrongly leaning posterior gets
its posterior logit pushed up hardest.

---

## 3. Focal-LogLTN for the band formulation

### 3.1 The mathematics

The published focal quantifier (Piano, Manigrasso, Russo, Morra, *Enhancing Neuro-Symbolic
Integration with Focal Loss: A Study on Logic Tensor Networks*, NeSy 2024, LNCS 14980,
[DOI 10.1007/978-3-031-71170-1_2](https://doi.org/10.1007/978-3-031-71170-1_2)) replaces the
mean-log universal aggregator with

$$
\log S_\gamma=\frac1N\sum_{i=1}^N \alpha_i\,(1-t_i)^\gamma\log t_i,
\qquad
L_\gamma=-\log S_\gamma=-\frac1N\sum_i (1-t_i)^\gamma\log t_i ,
$$

equivalently $S_\gamma=\prod_i t_i^{(1-t_i)^\gamma/N}$. The implementation uses
$\alpha_i = 1$. At γ = 0 this reduces **exactly** to §2.2. For γ > 0 the factor $(1-t_i)^\gamma$
suppresses already-satisfied atoms and concentrates gradient on violated or uncertain ones.

Instantiated on the two band formulas — inner truth $t=h_b(v)$, outer truth $t=1-h_b(v)$ so its
focal base is $h_b(v)$:

$$
\boxed{\;
L_{\mathrm{in},b}=-\frac1{n_{\mathrm{in},b}}\sum_{v\in B_{\mathrm{in},b}}\bigl(1-h_b(v)\bigr)^{\gamma_{\mathrm{in}}}\log h_b(v)
\;}
$$

$$
\boxed{\;
L_{\mathrm{out},b}=-\frac1{n_{\mathrm{out},b}}\sum_{v\in B_{\mathrm{out},b}} h_b(v)^{\gamma_{\mathrm{out}}}\log\bigl(1-h_b(v)\bigr)
\;}
$$

Numerically stable forms, with $-\log\sigma(r)=\operatorname{softplus}(-r)$:

$$
L_{\mathrm{in},b}=\frac1{n_{\mathrm{in}}}\sum(1-\sigma(r_H))^{\gamma_{\mathrm{in}}}\operatorname{softplus}(-r_H),
\qquad
L_{\mathrm{out},b}=\frac1{n_{\mathrm{out}}}\sum\sigma(r_H)^{\gamma_{\mathrm{out}}}\operatorname{softplus}(r_H).
$$

The side and patient reductions of §2.2 are unchanged.

**Gradient.** Writing $t=\sigma(s\,r_H)$ with $s=+1$ inside and $s=-1$ outside, and
$\ell_\gamma(t)=-(1-t)^\gamma\log t$:

$$
\boxed{\;\frac{\partial \ell_\gamma}{\partial r_H} = s\,(1-t)^\gamma\bigl[\gamma\, t\log t-(1-t)\bigr]\;}
$$

Two regimes matter.
- **Confidently wrong** ($t\to 0$): $\partial\ell_\gamma/\partial r_H \to -s$. A confidently
  wrong inner voxel keeps gradient ≈ −1 and a confidently wrong outer voxel ≈ +1. **The
  negative-log form retains a full corrective signal exactly where minimising a raw truth error
  like $1-t$ would die of sigmoid saturation.**
- **Confidently correct** ($e=1-t\to 0$):
  $|\partial\ell_\gamma/\partial r_H|\approx(\gamma+1)e^{\gamma+1}$ — raising γ suppresses easy
  voxels faster.

Distribution over the three logits: with $q_A=e^{z_A}/(e^{z_A}+e^{z_P})$, $q_P=1-q_A$ and
$g=\partial L/\partial r_H$, we get $\partial L/\partial z_A = g q_A$,
$\partial L/\partial z_P = g q_P$, $\partial L/\partial z_0 = -g$. It moves mass between
grouped foreground and background; it never expresses an A-versus-P preference.

**Asymmetric γ.** The code permits $\gamma_{\mathrm{in}} \neq \gamma_{\mathrm{out}}$. This does
not break $\mathcal G(\neg H)=1-\mathcal G(H)$; it applies *different focal aggregators to the
two formulas*,
$\phi_{\mathrm{in}}=\forall^{F_{\gamma_{\mathrm{in}}}}_{v\in B_{\mathrm{in}}}H(v)$ and
$\phi_{\mathrm{out}}=\forall^{F_{\gamma_{\mathrm{out}}}}_{v\in B_{\mathrm{out}}}\neg H(v)$.
It should be described as a **formula-specific asymmetric Focal-LogLTN extension**, not as the
simplest instance of one shared quantifier. Note also that a formula *weight* changes overall
strength while its *γ* changes how it distributes gradient between easy and hard voxels — they
are different knobs.

**Reporting caveat.** $S_\gamma$ is not idempotent: if every atom has the same truth
$t\in(0,1)$ then $S_\gamma=t^{(1-t)^\gamma}>t$. So `exp(-case_loss)` is a confidence-weighted
focal satisfaction diagnostic, **not comparable across γ values**. Compare γ settings on
segmentation metrics and error counts, never on this number.

### 3.2 The runs conducted before the 30-epoch runs

Every focal variant was screened as a **5-epoch pilot**, seed 0, bs 1, AMP on, warm-up 5,
`--constraint-eval-every 5`, each with its **own recalibrated λ** from the same frozen epoch-5
Dice-only checkpoint (protocol dir `focal1_protocol_20260829/`, launchers in
`cluster_protocol_20260830/`):

| run | $\gamma_{\mathrm{in}}$ | $\gamma_{\mathrm{out}}$ | epochs | Dice | Δ vs baseline @ep5 | Δ vs bands pilot |
|---|---:|---:|---:|---:|---:|---:|
| `msd_fold0_bands_focal_inner1_outer0_pilot_seed0_20260830` | 1 | 0 | 5 | **0.8566184** | **+0.0104627** | +0.0027443 |
| `msd_fold0_bands_focal05_pilot_seed0_20260830` | 0.5 | 0.5 | 5 | 0.8542144 | +0.0080587 | +0.0003403 |
| `msd_fold0_bands_focal_inner0_outer1_pilot_seed0_20260830` | 0 | 1 | 5 | 0.8539046 | +0.0077489 | +0.0000305 |
| *(reference)* `msd_fold0_bands_pilot_seed0_20260807_082555` | 0 | 0 | 5 | 0.8538741 | +0.0077184 | — |
| `msd_fold0_bands_focal1_pilot_seed0_20260829` | 1 | 1 | 5 | 0.8517742 | +0.0056186 | −0.0020999 |
| ~~`msd_fold0_bands_focal_inner05_outer0_pilot_seed0_20260830`~~ | 0.5 | 0 | 5 | **cancelled** | — | — |

The last one was `CANCELLED+` at 00:06:25 with only `config.json` and an empty `metrics.csv`
(6.5 KB, no checkpoints); it was **deleted 2026-09-03**. Its intended config is preserved:
bands, 5 epochs, bs 1, AMP on, fold 0, seed 0, γ_in 0.5 / γ_out 0, λ 0.01142961227755672. A
directory scan confirmed it was the only incomplete run in the project.

**The gate and the amendment.** `evaluation/gate_focal_pilot.py` rejected the winning
inner-1/outer-0 pilot **solely** on foreground false negatives: 9,622 against a 9,503 maximum —
over by 119 voxels. It had nonetheless improved hard Dice 0.853874 → 0.856618, reduced
foreground errors by 450, reduced total mislabelled voxels by 821, and produced a paired 95%
bootstrap CI of [+0.000596, +0.005047]. The user approved a **30-epoch exploratory extension**
(`protocol_amendment_inner_focal1_30epoch.md`, 2026-08-30), explicitly recorded as a
post-pilot amendment rather than a passed gate.

The 30-epoch runs that followed:

| run | config | epochs | λ | final | best | ep21–30 |
|---|---|---:|---|---:|---:|---:|
| `msd_fold0_bands_focal_inner1_outer0_30epoch_seed0_20260830` | γ 1/0 | 30 | 0.011383761103380578 | 0.8791155 | 0.8812316 (ep 16) | 0.8789536 |
| `msd_fold0_bands_focal_inner1_outer0_telemetry_30epoch_seed0_20260831` | γ 1/0 + telemetry | 30 | **[cluster-only]** | 0.879116 | 0.881232 (ep 16) | — |

The second is a **duplicate** — re-run only to add telemetry probes, and since training is
deterministic given the seed it has an identical trajectory. Counting both double-counts one
experiment.

### 3.3 The verdict on focal

At the **schedule-identical epoch 21–30 window** — the only exposure-fair comparison, since
StepLR decays at epoch 20 regardless of run length — against the official baseline:

| run | ep21–30 mean | Δ vs baseline |
|---|---:|---:|
| bands (BCE, γ = 0) | 0.8788906 | +0.0019266 |
| focal γ = 1/0 | 0.8789536 | +0.0019896 |
| class Tversky | 0.8787522 | +0.0017882 |

**Identical.** The apparent best-epoch "wins" (focal +0.000907 over bands, Tversky +0.000321)
come from selecting the maximum of a noisier curve:

| run | per-epoch plateau σ | plateau epochs | order-of-magnitude max-selection inflation |
|---|---:|---:|---:|
| none / translation / bands (50 ep) | ~0.0005 | 25 | ~0.001 |
| focal 30 ep | 0.001523 | 19 | ~0.004 |
| Tversky 30 ep | 0.001062 | 19 | ~0.003 |

The mechanism is in the project's own telemetry: focal and Tversky concentrate **94–99% of the
auxiliary gradient in the hardest 10% of band voxels**, raising gradient variance, which makes
the validation curve noisier, which inflates its maximum. It also explains the paradox that
focal has a higher best-epoch Dice with *more* total decoded errors (37,832 against bands'
37,513).

**And there was a prior reason to expect this.** §2.5 finding 3 — the band already achieves a
76× concentration of anti-FP pressure purely through its denominator. Focal is hard-negative
selection applied on top of a term that is already doing hard-negative selection.

---

## 4. The Tversky pilot, and why it could have beaten focal

### 4.1 What it is mathematically

`bands/class_aware_tversky.py`, `ClassAwareBoundaryTverskyLoss`. Two things change at once
relative to focal.

**(i) The bands become class-aware.** Instead of one band around the GT foreground *union*,
each foreground class $c\in\{A,P\}$ gets **its own** two-step inner/outer band built from its
own target mask $T_c=\{v: g(v)=c\}$:

$$
B_{\mathrm{in}}^{(c)} = T_c\setminus E_2(T_c),\qquad B_{\mathrm{out}}^{(c)}=D_2(T_c)\setminus T_c,
\qquad m_c = \mathbf 1_{B_{\mathrm{in}}^{(c)}} + \mathbf 1_{B_{\mathrm{out}}^{(c)}} .
$$

Because the anterior band and the posterior band each straddle the **internal A/P interface**,
that interface is now supervised. In the grouped formulation it is invisible — the union has no
interface.

**(ii) The voxelwise log-loss becomes a set-level Tversky index.** On the class band, with
$p_c = \mathrm{softmax}(z)_c$ and $y_c=\mathbf 1[g=c]$, summing over the spatial dimensions:

$$
\mathrm{TP}_c=\sum_v m_c p_c y_c,\quad
\mathrm{FP}_c=\sum_v m_c p_c(1-y_c),\quad
\mathrm{FN}_c=\sum_v m_c(1-p_c)y_c,
$$

$$
S_c=\frac{\mathrm{TP}_c+\epsilon}{\mathrm{TP}_c+\beta_{\mathrm{FP}}\mathrm{FP}_c+\beta_{\mathrm{FN}}\mathrm{FN}_c+\epsilon},
\qquad
L=\frac{1}{|\mathcal V|}\sum_{b\in\mathcal V}\frac12\sum_{c\in\{A,P\}}\bigl(1-S_c\bigr),
$$

with $\beta_{\mathrm{FP}}+\beta_{\mathrm{FN}}=1$ enforced, and the run using
$\beta_{\mathrm{FP}}=0.60$, $\beta_{\mathrm{FN}}=0.40$. A case is valid only if both classes
have non-empty inner and outer bands. At $\beta=(0.5,0.5)$ this is the Dice index; the
asymmetry is the whole point.

**Note for the LogLTN framing:** unlike bands and focal, this is **not** a LogLTN objective. It
is a $1-S$ loss over a *set-level* fuzzy score, not a $-\log S$ aggregation over per-voxel
atoms. It sits outside the logic framing of §2–§3, which is worth stating explicitly if the
thesis presents the boundary line as one neuro-symbolic family.

### 4.2 Why it was a reasonable bet against focal

Three independent arguments, all of which were live at the time:

1. **It attacks the one thing the grouped band provably cannot.** §2.5 showed the grouped
   log-odds distributes the "be foreground" gradient in proportion to current belief
   ($g\,q_A$, $g\,q_P$), so it *reinforces whichever class is already winning* and is
   structurally incapable of correcting a swap. Per-class bands supervise the A/P interface
   directly. And the geometry panel had just revealed that A/P assignment is the *only* thing
   the band actually improves — so making that mechanism explicit rather than accidental was
   the obvious next move.
2. **Asymmetry is a different knob from focal.** Focal reweights *within* a fixed
   error type by difficulty; Tversky reweights *between* error types by cost.
   $\beta_{\mathrm{FP}}=0.60 > \beta_{\mathrm{FN}}=0.40$ penalises false positives 1.5× harder,
   which is the correct direction given the baseline error budget (19,512 FP against 16,397 FN,
   FP being the larger group) and given that the focal-1/0 pilot had been rejected *precisely
   for pushing FN up*. Tversky is the natural instrument for exactly that failure mode.
3. **A set-level score has a different gradient geometry.** Tversky's denominator couples all
   band voxels of a class, so a voxel's gradient depends on the whole band's current TP/FP/FN
   balance rather than only on its own truth. In principle that avoids focal's variance
   problem: it can shift the operating point without concentrating gradient on individual hard
   voxels.

**The run:** `msd_fold0_bands_class_tversky_fp060_fn040_30epoch_seed0_20260901`, 30 epochs,
seed 0, bs 1, AMP on, warm-up 5, telemetry on (probe epochs 1 3 5 8 10 12 14 16 18 20 25 30),
λ from `bands_class_tversky_fp060_fn040_calibration.json` **[cluster-only]** — calibrated by
the same 10%/50% gradient-RMS ritual, but with `--bands-loss-type class_tversky`.

**The result:** final 0.8767429 (Δ +0.0007016), best 0.8806345 (Δ +0.0020355), ep21–30 mean
0.8787522 (Δ **+0.0017882**) — i.e. at matched exposure it is **indistinguishable from plain
BCE bands (+0.0019) and from focal (+0.0020)**.

**Why argument (1) failed in practice.** The objective was satisfied by *reallocating* the
swaps rather than removing them: anterior→posterior fell by 481 while posterior→anterior rose
by 660 — a net increase, with no boundary improvement. In neuro-symbolic terms this is a
textbook **reasoning shortcut**: the model found a way to satisfy the constraint through an
unintended mechanism. Framing it that way is stronger than reporting "another thing that did
not work", and it is corroborated by the same 2–3× curve-noise signature as focal. The paired
comparison gave **p = 0.993** on 25/52 cases.

---

## 5. The signed-distance one-cut constraint

### 5.1 The idea in words

The band asks a *per-voxel* question ("is this voxel inside or outside?"). The one-cut rule
asks a *profile* question: walk outward along the surface normal from a ground-truth boundary
face; you should cross from hippocampus to background **exactly once**, in that direction, and
near the annotated surface. That single statement constrains **existence, direction,
multiplicity and location** of the transition — four properties the band cannot see, because
the band is distance-blind (a voxel one step outside and one two steps outside receive
identical penalty).

### 5.2 Geometry — where the signed distance enters

`onecut/outer_onecut.py` and `Surface-normal ordinal LogLTN/surface_normal_ordinal/geometry.py`.

1. **Faces.** Take $F=\{v:g(v)>0\}$ and collect every face between a foreground voxel and a
   background neighbour, with its discrete outward direction. Face centres are at half-integer
   offsets.
2. **Signed distance field.** With physical spacing $\mathbf h$,

   $$\Phi(v)=\mathrm{EDT}_{\mathbf h}(\neg F)(v)-\mathrm{EDT}_{\mathbf h}(F)(v)$$

   — negative inside, positive outside, computed with `scipy.ndimage.distance_transform_edt`
   using `sampling=spacing`, so it is in **millimetres**, not voxels.
3. **Normals.** $\mathbf n = \nabla\Phi/\lVert\nabla\Phi\rVert$, sampled at face centres by
   trilinear interpolation of the gradient. Degenerate norms fall back to the discrete face
   direction, and any normal whose dot product with the face direction is negative is flipped —
   the discrete face direction is authoritative about inside-to-outside orientation, since
   numerical gradients misbehave at sharp corners.
   **The SDF is used only to obtain spacing-aware normals. It is never predicted by a new head
   and never appears in the loss.**
4. **Rays.** Offsets $o_0<\dots<o_{S-1}$ from $-3$ mm to $+3$ mm every $0.5$ mm (13 samples),
   with coordinates $\mathbf x_j = \mathbf p + o_j\,\mathbf n/\mathbf h$. Rays that would leave
   the volume are dropped; at most 4096 deterministic faces per patient (fixed geometry seed);
   the geometry is cached per label hash.

### 5.3 The logical formula

Sample the grouped semantic field of §2.2 along the ray, $u_j = r_H(\mathbf x_j)$ by trilinear
`grid_sample`. A **cut after sample $k$** denotes the complete assignment

$$
\Phi_k=\Bigl(\bigwedge_{j\le k}H(x_j)\Bigr)\wedge\Bigl(\bigwedge_{j>k}\neg H(x_j)\Bigr),
$$

and the formula existentially marginalises the uncertain crossing position over the admissible
cuts $C=\{k : |\tfrac12(o_k+o_{k+1})|\le \tau\}$ with tolerance $\tau = 1$ mm:

$$
\boxed{\;\Phi_{\mathrm{onecut}}=\bigvee_{k\in C}\Phi_k\;}
$$

The $\Phi_k$ are **mutually exclusive** assignments, so the disjunction is a genuine sum of
probability mass — evaluated stably in log space:

$$
\log T_k=\sum_{j\le k}\log\sigma\!\Bigl(\tfrac{u_j-m}{T}\Bigr)+\sum_{j>k}\log\sigma\!\Bigl(\tfrac{-u_j-m}{T}\Bigr)
$$

computed by prefix/suffix cumulative sums of `logsigmoid`, then

$$
\log T_{\mathrm{ray}}=\operatorname{logsumexp}_{k\in C}\log T_k,
\qquad
\ell_{\mathrm{ray}}=-\frac{\log T_{\mathrm{ray}}}{S},
\qquad
L_{\mathrm{onecut}}=\frac{1}{|\mathcal V|}\sum_{b\in\mathcal V}\overline{\ell_{\mathrm{ray}}}^{\,b}.
$$

Dividing by the number of ray literals $S$ makes the loss comparable across ray lengths; the
reported truth is $\exp(\log T_{\mathrm{ray}}/S)$, and a second diagnostic
$\exp(\log T_{\mathrm{ray}} - \operatorname{logsumexp}_{\text{all }k}\log T_k)$ gives the
*allowed* mass conditional on any one-cut assignment. Run settings: radius 3 mm, step 0.5 mm,
tolerance ±1 mm, margin $m=0$, temperature $T=1$, ≤4096 faces.

This is again a genuine LogLTN object: a **guarded universal over rays of an existentially
quantified conjunction**, optimised as negative log-satisfaction. It is the most expressive
logical statement in the project.

### 5.4 Why it should have worked better than the alternatives

1. **The predecessor failed for a diagnosable reason, and this fixes exactly that reason.** The
   original formulation was a *pairwise ordinal* rule,
   $t=\sigma\bigl((r_H(b-\delta n)-r_H(b+\delta n)-m)/T\bigr)$ — "inside scores higher than
   outside". Its audit on 139,916 rays: 27,532 erroneous rays (19.68%), only 4,066 ordinal
   violations (2.91%), **error coverage 14.77%** at **100% precision** (lift 5.08×). Perfectly
   specific, almost blind. The problem was **coverage, not gradient direction**: most boundary
   errors are *shifted, missing or multiply-crossing* while still respecting the local ordering
   at ±1 mm. The one-cut formula was designed precisely to see those.
2. **And it does see them, overwhelmingly.** Same cases, same frozen logits:

   | interface | ROC AUC | error coverage @5% correct-ray FPR | precision | lift |
   |---|---:|---:|---:|---:|
   | outer H/background | **0.99996** | **100.0%** | 83.05% | 4.22× |
   | anterior/posterior | 0.99395 | 98.24% | 93.62% | 2.18× |

   (A structural sanity check, not independent predictive evidence — the error categories and
   the formula are defined from related ray-crossing conditions.)
3. **It is physically distance-aware.** The band is distance-blind by construction. One-cut
   works in millimetres via spacing-aware normals and a physical tolerance, which is the
   property the canonical boundary-loss family (Kervadec; Karimi & Salcudean) provides and
   which was otherwise entirely absent from this project.
4. **Its frozen-logit repair passed the pre-registered screen.** `Dice + outer one-cut` against
   an equal-RMS `Dice only` update: union Dice **+0.000196**, 1 mm surface Dice **+0.000533**,
   2 mm surface Dice +0.000097, ASSD **−0.000670 mm**, HD95 **−0.01249 mm**; improved in
   **61.5%** of cases, worse in 5.8%, unchanged in 32.7%; FP −0.10 and FN −1.12 voxels. The
   Dice-gradient cosine was positive in *every* case (mean ≈ 0.107) and the per-case weight for
   a 10% RMS match had median ≈ 0.0198. Verdict: **`GO_TO_SHORT_PILOT`**.
5. **It is not redundant with the band.** The hybrid feasibility audit measured median
   band/one-cut gradient cosine **+0.4097** with **91.2%** of the one-cut gradient surviving
   projection onto the band gradient. The two terms genuinely carry different information.

The honest counterweight, recorded at the time: the same audit did **not** show one-cut
superior to the band. One-cut led on 1 mm surface Dice by +0.000130, but bands led on union
Dice by 0.000639 and on ASSD by 0.002724 mm, and reduced FP/FN more consistently (one-cut
higher 1 mm surface Dice in 36.5% of cases, lower in 44.2%). The A/P variant was deferred: its
direct update cut swaps strongly, but at the 10% matched scale it gave A/P swaps −0.423 voxels
against union Dice −0.000044 and 1 mm outer surface Dice −0.000038 — a real objective
trade-off, so it was excluded from the outer pilot.

### 5.5 The runs conducted before it was excluded

**Protocol** (`docs/cluster_protocol_20260902/README.md`), a fixed three-step sequence:

| # | run | what | epochs | seed |
|---|---|---|---:|---:|
| 1 | `onecut_protocol_20260902/msd_fold0_none_calibration_seed0` (job 646126) | fresh Dice-only control from scratch, current source `cf7d77bc…` | 5 | 0 |
| 2 | `calibrate_onecut.sbatch` | frozen-checkpoint one-cut gradient calibration, 32 deterministic training cases, 10% target / 50% p95 cap → `onecut_calibration_seed0.json` | — | 0 |
| 3 | `onecut_protocol_20260902/msd_fold0_onecut_pilot_seed0` | matched one-cut arm from scratch | 5 | 0 |

Everything else matched: MSD fold 0, saved split, 64³ crops, AMP, AdamW lr 1e-4, warm-up 5,
same A100 partition, λ read from the calibration report and re-validated by the trainer.

**Result:**

| arm | Dice @ep5 | Δ vs matched control | Δ vs bands pilot |
|---|---:|---:|---:|
| Dice-only control | 0.8461556 | — | — |
| **one-cut pilot** | **0.8534984** | **+0.0073428** | **−0.0003757** |
| bands (BCE) pilot | 0.8538741 | +0.0077184 | — |

**It was excluded for losing to bands by 0.00038 hard Dice.** That decision does not stand up:

- 0.00038 is **below the 0.0004 same-seed reproducibility floor**, and **34× below the 0.0129
  seed noise measured at 5 epochs**.
- The gate fired during **warm-up** — with `α_e = min(1, e/5)`, the pilot saw exactly one epoch
  at full constraint strength.

**But an independent problem was found later, and it is real.** The discovery phase had a good
habit — measure how often the *ground truth* satisfies a rule before adopting it (260/260 for
posterior connectedness, 260/260 for elongation). The one-cut rule never got that test. It was
run locally on the 52 fold-0 validation labels using the repository's own
`OuterOneCutLogLTNLoss._build_rays` geometry:

**139,916 rays, ground truth only, at the protocol's 3 mm radius:**

| outcome | rays | share |
|---|---:|---:|
| valid single cut inside ±1 mm tolerance | 130,244 | **93.09%** |
| **two or more transitions (ray exits and re-enters)** | **9,650** | **6.90%** |
| single cut outside tolerance | 14 | 0.01% |
| zero transitions | 6 | 0.00% |
| single cut, wrong direction | 2 | 0.00% |

Per-case: mean 0.9314, min 0.8516, **11 of 52 cases below 90%**. At radius 2 mm satisfaction is
98.12%; at 1.5 mm, 98.95%.

**So the rule is false on its own ground truth for about one ray in fourteen**, and the
tolerance interval is not the cause (0.01%) — multi-transition rays are. The mechanism is
thin structure: only 29.5% of the hippocampus survives two erosions, so a 3 mm inward ray
frequently exits through the opposite surface, making the true profile FG→BG→FG, which no
single cut can describe. Wherever that happens, the constraint pushes toward a configuration
the label itself violates.

**Two hybrids were also audited and both returned NO_GO before any training:**

| audit | key numbers | gates | verdict |
|---|---|---|---|
| `hybrid_feasibility_20260902` (BCE band + one-cut) | band/one-cut cosine +0.4097, residual 0.912, anchor/location cosine +0.8262, residual 0.563, simultaneous descent in 100% of cases | 3 of 4 PASS; **FAIL** `matched_counterfactual_improvement` | **NO_GO** — at every update RMS (0.02/0.05/0.1) union Dice fell (−0.000442 / −0.000978 / −0.001595) and total errors per case rose (+2.98 / +6.90 / +11.44), buying only 1 mm surface Dice |
| `band_location_sweep_train32_20260902` (band-dominant + cut-location residual) | 32 training cases, 0 skipped, median band/location cosine +0.4526, residual 0.892 | **0 of 12** configurations passed (4 location shares × 3 update magnitudes) | **NO_GO** — surface Dice gains of 2e-5…8e-4 always came with negative union Dice and more errors per case |

**Current standing:** the one-cut rejection is **not safe**, but the formulation that was
rejected was also **partly invalid**. The fair re-test is at a corrected radius (2 mm or
1.5 mm) and at converged exposure — it has never been run. A separate consideration weighs
against reviving it for A/P: the prediction *already* satisfies single-crossing internally (one
connected interface component in 43/52 cases, only 189 voxels total across all 52 cases outside
the largest component of their class), so an ordering/one-cut/connectedness constraint has
essentially no gradient to give there.

---

## 6. Current direction — translation equivariance

### 6.1 The rationale, plainly

Dice measures overlap; it says nothing about *stability*. If the crop shifts two voxels, the
hippocampus has not changed — so the segmentation should shift by exactly the same two voxels:

$$f(T x)\approx T f(x).$$

A large change under a small shift signals positional dependence or internal instability that
voxelwise supervision never penalises. The appeal is that it needs **no atlas, no
registration, no average shape**, imposes nothing subject-specific, and produces a dense
differentiable signal. In the initial audit, violation of this rule correlated with larger
segmentation error (K1 in the discovery phase: violation/error correlation **0.539**, the
highest of any candidate rule, violated by 30/52 predictions).

There is a strong caveat that must be in the thesis: **this pipeline has no data augmentation
at all**, so the equivariance arm is the only mechanism that ever shows the model a translated
input. It differs from its control in three ways at once — augmentation (it sees shifted
inputs), consistency (it is penalised for disagreeing on the *same* input), and compute (a
second forward pass per step). Separating these is the highest-value open control.

### 6.2 What the latest run and the current batched job actually optimise

Both use `--constraint-set equivariance`, i.e. `TranslationEquivarianceLoss` in
`thesis/new_constraints/equivariance/translation_equivariance.py`. Per training step:

1. Sample one shift $s$ uniformly from the six axis-aligned translations
   $\{(\pm2,0,0),(0,\pm2,0),(0,0,\pm2)\}$ (`--translation-size 2`), from a dedicated generator.
2. Forward the shifted image, softmax, and map back:
   $q = T_s^{-1}\,\mathrm{softmax}\bigl(f(T_s x)\bigr)$; $p=\mathrm{softmax}(f(x))$.
   Translation is exact integer slicing with zero padding — no interpolation.
3. Build the round-trip validity mask $m = T_s^{-1}T_s\mathbf 1$, which zeroes the rim lost to
   the shift.
4. For each hippocampal class $c\in\{1,2\}$, a **squared-denominator soft Dice** between the
   two probability maps:

$$
\boxed{\;
S_c=\frac{2\sum_v m\,p_c q_c+\varepsilon}{\sum_v (m\,p_c)^2+\sum_v (m\,q_c)^2+\varepsilon}
\;}
$$

5. Truth and loss:

$$
\mathrm{truth} = \tfrac12(S_1+S_2)\ \text{clamped to }[0,1],
\qquad
\boxed{\;L_{\mathrm{eq}} = 1-\overline{\mathrm{truth}}\;}
$$

6. Total: $L = L_{\mathrm{Dice}} + \min(1,e/5)\cdot 0.1\cdot L_{\mathrm{eq}}$.

The quadratic denominator makes the measure **reflexive**: two identical soft maps score
exactly 1. A second, **linear**-denominator Dice is also computed and logged as
`confidence_weighted_agreement` with a 0.90 adherence threshold — this is the legacy metric and
it is *confidence-sensitive*: identical but uncertain maps can score below 1. It measures
consistency and confidence simultaneously and must not be presented as standalone evidence of
equivariance. The reported "adherence" percentage is over **case × translation pairs**
(52 × 6 = 312), not over patients satisfying all six directions.

**Two details specific to the current Family-B jobs.** `--equivariance-max-samples 1` with
`--batch-size 1` means the extra translated forward is applied to the single sample of every
batch. `--constraint-eval-every 0` disables periodic constraint evaluation during training;
the exhaustive six-translation measurement happens once at the end.

### 6.3 Can this be put into the LogLTN framework? Yes — and it should be

**Short answer:** the *grounding* is already fine; the *aggregation* is not. The constraint is
currently a $1-S$ loss over a **single case-level fuzzy score**, whereas bands, focal and
one-cut are all $-\log S$ losses over **per-atom truths**. Three levels of conversion, in
increasing fidelity:

**Level 1 — minimal, log-space only.** Keep the same grounding and read $S_{\mathrm{eq}}$ as
the truth of one grounded atom $\mathrm{Equivariant}_s(X_b)$. Then

$$L_{\mathrm{eq}}^{\log}=-\log S_{\mathrm{eq}} \quad\text{instead of}\quad 1-S_{\mathrm{eq}} .$$

This is a one-line change and it fixes the gradient asymmetry the bands document already
argues for: $\partial(1-t)/\partial t=-1$ everywhere, whereas
$\partial(-\log t)/\partial t=-1/t$ grows as the formula is more violated. It does **not** use a universal quantifier, so
call it a single-atom LogLTN term, not a LogLTN aggregation.

**Level 2 — proper guarded universal over voxels.** Ground a per-voxel binary predicate.
The natural choice is the **agreement probability**

$$
\mathcal G_\theta\bigl(\mathrm{Agree}_s(v)\bigr)=t_s(v)=\sum_{c} p_c(v)\,q^{(s)}_c(v)\in(0,1],
$$

the probability that independent draws from $p(v)$ and $q(v)$ give the same class. It equals 1
iff both are the same one-hot, is differentiable, and needs no new parameters. Then the guarded
universal over the valid support $V_s=\{v : m_s(v)=1\}$ is

$$
\log S_s=\frac{1}{|V_s|}\sum_{v\in V_s}\log t_s(v),
\qquad
\phi_s=\forall_{v\in V_s}\mathrm{Agree}_s(v),
$$

and equivariance over all six directions is a conjunction of the six formulas, which in log
space is their mean:

$$
\boxed{\;
L_{\mathrm{eq}}^{\mathrm{LogLTN}}
=-\frac{1}{|S|}\sum_{s\in S}\frac{1}{|V_s|}\sum_{v\in V_s}\log\Bigl(\textstyle\sum_c p_c(v)q^{(s)}_c(v)\Bigr)
\;}
$$

This is structurally **identical** to the bands objective — guarded universal, mean negative
log-satisfaction, geometric aggregation across formulas and cases — so the whole thesis would
present one aggregation scheme instead of two. It also concentrates gradient on the voxels that
actually disagree, instead of diluting it across the whole crop through a global Dice
denominator.

**Level 3 — Focal-LogLTN.** Reuse the §3 quantifier unchanged:

$$
L_{\mathrm{eq}}^{\gamma}=-\frac{1}{|S|}\sum_{s}\frac{1}{|V_s|}\sum_{v\in V_s}\bigl(1-t_s(v)\bigr)^{\gamma}\log t_s(v).
$$

Since equivariance disagreement is extremely sparse (final continuous truth ≈ 0.958), the
focal factor is arguably *more* justified here than it was on the bands, where the guard domain
was already doing hard-negative selection.

**Three things the conversion is worth doing for, and one it does not fix.**

- The recorded pathology in the current formulation is that the **continuous truth falls
  monotonically over training** (0.9977 → 0.9563 train; 0.9969 → 0.9593 val) while thresholded
  adherence rises to saturation. That is a direct symptom of a confidence-sensitive,
  case-level Dice-style score: early diffuse probabilities agree trivially, and as the model
  sharpens the same geometric disagreement costs more. A per-voxel log-truth does not have that
  confounding.
- $-\log t$ retains an order-one corrective gradient on confidently disagreeing voxels, where
  $1-S$ does not — the same argument the bands document makes in §13, and the same argument the
  GPU audit confirmed empirically for the KL teacher (§6.5).
- It makes the constraint reportable in the same table as bands, focal and one-cut, with
  comparable satisfaction semantics.
- **It does not fix the fixed-point problem.** All of Levels 1–3 are *symmetric* in $p$ and
  $q$: a consistently wrong, translation-invariant predictor satisfies them perfectly. That is
  intrinsic to a pairwise consistency relation and is exactly why the stop-gradient teacher
  (§6.6) exists as a separate candidate — though note the teacher does not escape it either
  (when $p=q$ its gradient vanishes too).

### 6.4 The equivariance runs and their results

**Family A (bs 2, AMP off, 50 epochs, seed 0)** — valid inside its own family only:

| pair | control best | equivariance best | Δ best | Δ final |
|---|---:|---:|---:|---:|
| era 0730 (`…none_20260730_124318_611097` vs `…translation_20260730_121523_611071`) | 0.874906 | 0.882354 | **+0.007448** | +0.006423 |
| era 0805/06 (`…none_20260806_110334_616958` vs `…translation_20260805_210911_616615`) | 0.874524 | 0.882494 | **+0.007970** | +0.009078 |

The effect is stable across the whole second half of training, not a peak artefact:

| window | era 0730 | era 0805/06 |
|---|---:|---:|
| ep 11–20 | +0.004014 | +0.005703 |
| ep 21–30 | +0.006950 | +0.007946 |
| ep 31–40 | +0.006707 | +0.008228 |
| **ep 41–50** | **+0.007158** | **+0.008934** |

Full record of the first isolated run (`docs/allenamenti.md`): Slurm **611071**, `swin-trans-f0`,
COMPLETED 0:0 in 00:27:54, λ_eq 0.10 hand-set, 50 epochs, warm-up 5, `--no-amp`, seed 0. Final
hard Dice 0.8793, best 0.8824 (ep 21), final optimisation satisfaction over all six
translations 0.9622, legacy linear satisfaction 0.9295, legacy adherence at 0.90 = 0.9391
(293/312 case-translation pairs). *(The logged `val_dice_soft = 0.0127` in that run is
invalid — the inherited MONAI evaluator booleanised positive softmax probabilities. It does not
affect the training loss or the hard Dice.)*

**Tuning asymmetry, in equivariance's favour:** bands received a gradient-RMS-calibrated
λ = 0.011506; equivariance received a **hand-set 0.1**. The less-tuned constraint is the one
with the larger effect, which pre-empts the "it just got a luckier weight" objection.

**Family B replication — `experiments/equivariance_family_b_20260905/`.** The point of this
experiment is to move the project's best training-side result into the official configuration.

| item | value |
|---|---|
| cluster root | `/mnt/beegfsstudents/home/3160552/equivariance_family_b_20260905_01` |
| source digest | `cf7d77bc714d99d96d9975eb84bfb0d38644cc6b7305e3defb017371953628bf` — **file-for-file identical to both official control manifests**; no CE, teacher, augmentation or diagnostic code enters these jobs |
| treatment | legacy foreground squared soft-Dice equivariance, λ 0.1, one extra translated sample per batch, uniform six axis shifts ±2, five-epoch linear warm-up |
| everything else | fold 0, bs 1, AMP, Dice including background, 64³, **no augmentation**, AdamW lr 1e-4 / wd 1e-5, StepLR 20 / γ 0.5, 50 epochs, final-only constraint evaluation, A100 MIG 4g.40gb |
| **seed 0** | Slurm **650078** — COMPLETED, exit 0, **19:25:44**, 50/50 epochs |
| **seed 1 (the current batched job)** | Slurm **650080** — started 2026-09-06 08:30:29 CEST; at the 11:26 check it had completed **epoch 7**; expected finish ≈ 04:00 on 2026-09-07 against a 24 h limit ending 08:30 |
| cancelled | 650079 (a reserved continuation) — cancelled to free a queue slot; the account allows 2 submitted / 1 running |
| resume path | `run.sbatch SEED PREVIOUS_JOB` restores model, optimizer, scheduler, AMP scaler and RNG from the atomic latest checkpoint; it retries **only** TIMEOUT / PREEMPTED / NODE_FAIL and skips a completed run |
| primary endpoint | paired mean hard-Dice difference at **epochs 41–50** against the same-seed official control; epoch 50 and mean 21–30 co-reported; best epoch exploratory |

**Seed-0 result** (`INTERIM_RESULTS_20260906.md`), paired against the exact seed-0 official
Dice control:

| endpoint | control | equivariance | paired Δ |
|---|---:|---:|---:|
| epochs 21–30 mean | 0.876964 | 0.882148 | +0.005184 |
| **epochs 41–50 mean (primary)** | 0.875515 | 0.881572 | **+0.006057** |
| epoch 50 | 0.874762 | 0.881510 | **+0.006748** |
| best epoch (exploratory) | 0.878599 @16 | 0.884449 @12 | +0.005850 |

**The paired delta is positive at every epoch from 6 through 50.** The primary effect is about
**5× the measured 0.001207 mean absolute seed gap** and about **2.5× the conservative 2× gap
reference**. Constraint health at epoch 50: train constraint loss 0.004444; continuous train
equivariance truth fell 0.996884 → 0.955556; final exhaustive validation truth 0.958434;
confidence-weighted agreement 0.943361; **threshold adherence 1.0 — saturated, and therefore
not interpretable as perfect continuous equivariance.**

**Seed-1 health at epoch 7:** all values finite, training loss 0.701876 → 0.126094, warm-up
complete and constraint active; epoch-7 validation Dice 0.867172 against 0.871871 in its matched
control, mean through epoch 7 **+0.002056**. Early-epoch seed noise (0.048 over epochs 1–5) is
far too large for either number to mean anything yet.

**The decision is still pending and is pre-specified:** accept transfer of the equivariance
effect only if the two seeds agree in direction and the two-seed mean at epochs 41–50 is
material relative to the observed control seed gap. A seed-1 sign flip makes the result
unstable — do not average it away, and do not substitute best epochs. (This is precisely the
discipline that caught the bands result.)

### 6.5 Supporting evidence from the frozen-logit GPU audit (Slurm 650074)

On the official epoch-50 checkpoint, 52 validation cases, AMP inference with float32
diagnostics, per-case-averaged macro Dice:

| prediction | macro Dice | Δ | better cases | FP | FN | swaps |
|---|---:|---:|---:|---:|---:|---:|
| official baseline, re-evaluated | 0.874766 | — | — | — | — | — |
| **full 12-shift teacher** | 0.881712 | **+0.006946** | 41/52 | −1,418 | −919 | +8 |
| identity + 12 shifts (13-view TTA) | 0.881602 | +0.006837 | 43/52 | −1,385 | −892 | −16 |
| sampled two-shift teacher | 0.876187 | +0.001421 | 29/52 | −1,218 | +57 | +320 |

The full translated ensemble removes **2,329 errors (5.93%)**, almost entirely at the outer
foreground boundary — which is the boundary-localisation hypothesis motivating the whole
translation line, and which also refutes any claim that the outer boundary is irreducible. Two
cautions attached to it: this is an *inference* result, not a measured training gain; and the
teacher's *incremental* frozen-logit repair over Dice alone is small (+0.000341 at the largest
tested step; +0.000440 over Dice+CE).

The audit also established the saturation numbers that constrain every smooth constraint:
max-class confidence ≥ 0.99 in **94.64% of GT foreground** and **90.72% of the multiclass GT
boundary**; **79.07%** of incorrectly decoded voxels are saturated, at mean confidence 96.69%.
But it simultaneously **refuted the universal version of that argument**: full-teacher KL alone
improves frozen logits at every tested step (+0.000249 / +0.001431 / +0.004739 at max logit
change 0.25 / 1 / 4). A stable logit-space loss retains corrective gradient on confident errors;
what fails is a loss whose derivative with respect to *probabilities* is bounded. **This is the
strongest argument for the LogLTN conversion in §6.3** — $-\log t$ is such a loss, $1-S$ is not.

### 6.6 Prepared but not submitted

From `NEXT_PROTOCOL.md`, staged at
`/mnt/beegfsstudents/home/3160552/translation_followup_20260905_01/source` (development digest
`8b30d57574dda9c89c52a96b63ad986d19ff2d97fd18e1a5e3e86d91825fe7b1`, 43 files, verified
identical locally and on the cluster; the running replication source is never edited):

1. **Augmentation-only control** (`--translation-augmentation`): identity with probability ½,
   otherwise one of the six ±2 axis translations uniformly; image and label move together with
   zero padding; validation unaugmented; sampling is a stateless function of seed/epoch/batch,
   separate from data-order and constraint RNG, so enabling a constraint cannot change the
   augmentation sequence. **This is the control that separates "consistency" from "finally
   showing the model a translated input"**, and it is the single most important missing
   experiment in this direction. It matches optimizer steps but **not** GPU work or two-view
   input exposure — a compute-matched arm is a further step.
2. **Augmentation + the same equivariance loss**, as the matched comparison against (1).
3. **Stop-gradient translation teacher** (`--teacher-support common`): with
   $q=\mathrm{stopgrad}\bigl(\mathrm{mean}_s\,T_s^{-1}f(T_s x)\bigr)$ and case KL
   $\frac{T^2}{|V|}\sum_{v}\sum_c q_c\log(q_c/p_c)$, the student-logit gradient at $T=1$ is
   $(p-q)/N$ — and on fixed common support, uniform two-view sampling has the **same expected
   gradient** as the full 12-view ensemble (verified by an exact 66-pair unit test). Common
   support is the intersection of valid overlap across all 12 configured shifts (±1, ±2 per
   axis), i.e. the central 60³ region = **82.397%** of crop voxels; the rim still gets ordinary
   Dice but no teacher KL. Requires the CUDA calibrator first (32 deterministic *training*
   cases from the verified epoch-5 Dice control; λ fixed by
   $\min(0.10/\mathrm{median(ratio)},\,0.50/p_{95}(\mathrm{ratio}))$), which **has not run on
   GPU yet**.

**Verification state as of 2026-09-05:** 219 distinct tests passed with one intentional skip;
scoped `git diff --check` passed; launchers checked with `bash -n`. These use synthetic tensors
and tiny models — they are not a GPU audit, and the revised source has not completed a real
SwinUNETR training smoke.

---

## 7. Where the project stands, and what I could not verify

### 7.1 Every direction, on one line each

| direction | matched, converged effect | verdict |
|---|---:|---|
| BCE bands (LogLTN, γ = 0) | **+0.000144** two-seed mean at ep 50, sign flips | does not replicate |
| Focal-LogLTN bands (γ = 1/0) | +0.0020 at ep 21–30, single seed | indistinguishable from bands |
| Class-aware boundary Tversky | +0.0018 at ep 21–30, single seed | indistinguishable from bands |
| One-cut LogLTN (SDF rays) | rejected at 5 epochs by 0.00038 | **rejection invalid**, but the rule is also false on 6.9% of GT rays at 3 mm |
| Band + one-cut hybrids | — | NO_GO before training, on frozen-logit audits |
| **Translation equivariance** | Family A **+0.0074 / +0.0080**; Family B seed 0 **+0.006057** (ep 41–50) | **the only real effect**; seed-1 replication in flight |

For calibration of scale: eliminating all A/P swaps is worth **+0.0231**; translation TTA at
inference is worth **+0.0093 to +0.0095** for zero training cost; and the recipe gap to nnU-Net
(~0.889 on this task, with full augmentation, deep supervision and ensembling, against this
baseline's 0.8748) is 10–18× any constraint effect measured here. That gap is an uncontrolled
alternative explanation for anything attributed to a constraint — **especially** equivariance,
which is the only mechanism in this pipeline that ever shows the model a translated input.

### 7.2 Verification notes

Everything above was read from the repository at `~/Desktop/hippo`, principally:
`docs/allenamenti.md`; `docs/RESEARCH_CRITIQUE_20260905.md`; `docs/REPO_AUDIT_20260905.md`;
`experiments/equivariance_family_b_20260905/{SUBMISSION.json, VERIFICATION.md,
INTERIM_RESULTS_20260906.md, NEXT_PROTOCOL.md, run.sbatch, next_training.sbatch}`;
`experiments/loss_constraint_followup_20260905/{GPU_AUDIT_RESULTS_20260905.md, PROTOCOL.md}`;
`cluster_protocol_20260830/*.sbatch` and `protocol_amendment_inner_focal1_30epoch.md`;
`cluster_protocol_20260902/{README.md, *.sbatch, hybrid_feasibility_20260902/REPORT.md,
band_location_sweep_train32_20260902/REPORT.md}`;
`thesis/new_constraints/bands/{Focal Log-LTN.md, SCIENTIFIC_RATIONALE_AND_FORMULATION.md,
class_aware_tversky.py, README.md}`; `thesis/new_constraints/onecut/*`;
`thesis/new_constraints/equivariance/*`; `docs/thesis/new_constraints/teacher/README.md`;
`thesis/Surface-normal ordinal LogLTN/{README.md, EXPLORATORY_FOLD0_RESULT.md,
surface_normal_ordinal/*}`; and the worktree
`.claude/worktrees/hippo-boundary-constraints-5f4b77/{OFFICIAL_BASELINE_20260903.md,
COMPARABILITY_REGISTER_20260903.md, PREREG_BANDS_REPLICATION_20260902.md,
BOUNDARY_LINE_CRITIQUE_20260902.md, RESULT_POOL_ANALYSIS_20260902.md,
SESSION_HANDOFF_20260903.md, PROJECT_PROMPT_20260905.md, NEW_DIRECTIONS_20260902.md}`.

**Not re-verified from primary run artifacts.** No `metrics.csv`, `final_metrics.json`,
`config.json` or `completion_manifest.json` exists anywhere in the local repository — every
per-epoch trajectory lives only under `/mnt/beegfsstudents/home/3160552/`. The numbers above
are as recorded in the project's own dated documents, which are internally consistent (e.g. the
official baseline's epoch-50 value 0.8747625 appears identically in four independent documents
and re-evaluates to 0.8747659 through the current evaluator, a 3.5e-6 difference).

**The cluster could not be reached from this session.** `bocconi-cluster` is an alias in your
Mac's `~/.ssh/config`; this session's sandbox has network access but neither that config nor
your SSH keys, so the name does not resolve. That said, nothing appears to be missing: the
Family-B interim results file was last written at **11:53 CEST today**, three minutes before I
started reading it, so the local record already reflects the current cluster state. The only
thing a live login would add right now is seed 1's current epoch — it was at epoch 7 as of the
11:26 check and is not expected to finish until ~04:00 tomorrow.

**Values that exist only on the cluster** and would be worth pulling when you next log in:

- per-epoch `metrics.csv` for the seed-1 control (648443) and seed-1 bands (648444);
- λ for the telemetry focal run (`bands_focal_inner1_outer0_telemetry_v2_calibration.json`) and
  for class Tversky (`bands_class_tversky_fp060_fn040_calibration.json`);
- `recommended_onecut_weight` from `onecut_calibration_seed0.json`;
- the seed-1 equivariance trajectory (650080) once it completes.
