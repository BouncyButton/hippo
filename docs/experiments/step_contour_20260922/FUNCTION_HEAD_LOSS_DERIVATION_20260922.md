# Function head with voxel feedback: loss derivation (22 Sep 2026)

Status: design specification, not an implemented or validated model. The current fold-0 outer validation has informed the design and is exploratory; confirmatory evaluation needs a locked, previously untouched split or nested cross-validation.

**Revision 2 (22 Sep 2026, after design review).** Changes from revision 1:
(1) the conditional edge loss pushed true interior foreground toward background (gradient `m·p_z/π > 0`); replaced by a miss bin with **one shared presence** for both edge heads, whose presence NLL counts once per head; (2) presence is trained on rays outside the ground-truth box too; (3) new arm D′ (trained without rendering, rendered only at inference) so that E vs D′ tests training *through* the renderer; (4) anchor compression for rescue is an **experiment** gated on measured log-hit gaps, not a settled fix; (5) thin-ray rendering, the warm-up loss normalization, the strictly positive solver ridge, and the split of the curve-loss gradient into anchor and residual paths are specified. Voxel-gradient claims were re-checked numerically on toy columns (numpy finite differences).

Notation:

- x is the sagittal slice, y the longitudinal position, and z the ray axis (z = 0,…,Z−1).
- A **ray** r = (x, y) is one column along z.
- The network outputs 3-class logits (ℓ_B, ℓ_A, ℓ_P) per voxel: background, anterior, posterior.

## 0. Ground-truth functions (exact, never fitted)

From the union foreground label y*_z ∈ {0,1} on each ray:

- presence q*_r = 1[∃z: y*_z = 1]
- lower edge L*_r = min{z : y*_z = 1}, upper edge U*_r = max{z : y*_z = 1}
- for absent rays, set L* = Z and U* = −1 (this makes the sets in §3 cover the whole ray)

For fixed x, L*(y) and U*(y) are already exact integer step functions. Fitting a step model to them would erase the valid upward jumps (12/52 references).

`L_resp` and the edge terms use rays inside a dilated slice bounding box, so that thousands of trivially empty rays do not dominate. The **presence** term is the exception: it is trained on all rays, with out-of-box rays down-weighted or subsampled (§2).

## 1. Differentiable first- and last-hit distributions (voxel path, not detached)

The foreground logit is a_z = logsumexp(ℓ_A, ℓ_P) − ℓ_B, with p_z = σ(a_z).

```
first hit   f_r(z) = p_z · Π_{t<z} (1 − p_t)
last hit    l_r(z) = p_z · Π_{t>z} (1 − p_t)
miss        m_r    = Π_t (1 − p_t),     voxel presence π_r = 1 − m_r
```

Compute everything in log space: log p = logsigmoid(a), log(1−p) = logsigmoid(−a), then an exclusive cumsum (a reverse one for l). This is O(Z), runs on the GPU and has no Python loop.

**Why a miss bin.** Revision 1 used `φ = softmax_z(log f + g)`. At g = 0 its edge CE is `−log f(L*) + log π`, the first-hit CE *conditional on a hit*. The `log π` term has gradient `m·p_z/π > 0` on every voxel's foreground logit, including true interior voxels after L*, so it pushes correct foreground toward background on uncertain rays. Numerical check, GT foreground z = 3–6: interior gradient +0.055 (interior p = 0.5) and +0.13 (nearly empty ray); 0 with the unconditional loss.

**Head input.** Treat each sagittal slice as a 2-D (y, z) image with channels [p, f, l, projected decoder features]. A small 2-D CNN predicts residual edge logits g^L(y,z), g^U(y,z) and a residual presence logit g_π(y). Zero-initialize all three final layers.

**Shared-presence factorization.** From the log-space hit code:

```
log m = Σ_t log(1 − p_t)                    log π = log(−expm1(log m))      (voxel presence)
A^L(z) = log f(z) − log π                   A^U(z) = log l(z) − log π       (conditional log hit mass)
π̂ = σ(log π − log m + g_π)                                                  (ONE presence per ray)
φ^L = softmax_z(A^L + g^L)                  φ^U = softmax_z(A^U + g^U)       (edge | occupied)
P^L(z) = π̂ · φ^L(z),  P^L(∅) = 1 − π̂        P^U(z) = π̂ · φ^U(z),  P^U(∅) = 1 − π̂
```

At g = 0: `P^L(z) = f(z)`, `P^U(z) = l(z)` and `P(∅) = m` exactly (checked: max error 1.7e-16). The lower and upper heads cannot disagree about whether the ray exists. Compute log π with `expm1` from log m, never as log of a rounded probability.

**Rescue and the anchor range (experiment, not a settled fix).** log-hit gaps between the true edge and the mode of A can be tens of nats, which a zero-initialized residual may not learn to overcome. A hard floor (e.g. at −8) would break the exact initial tie and cut the anchor gradient below the floor, so it is not adopted. Gate:

1. From a baseline checkpoint, measure the gap `d = max_z A(z) − A(L*)` (and for U) on occupied training rays; report its distribution and how often the GT edge lies below any candidate floor.
2. If rescue is needed, compare two options:
   - **rescue mixture (preferred):** `φ = (1−α)·softmax(A+g) + α·softmax(h)`, with h a feature-only edge map and α = σ(α_raw + g_α) initialized near e⁻⁶. The initial tie is exact up to α, the anchor gradient is intact, and the rescue range does not depend on the gap. Compute in log space with `logaddexp`.
   - **smooth compression:** `Ã = −κ·tanh(−A/κ)`, which is ≈ A near 0, bounded at −κ, and never has zero gradient; it perturbs the tie only where A ≪ 0.

**Diagnostics.** Report the centered residual magnitude `mean_z |g(z) − mean_z g|` on occupied rays, |g_π|, and the distance between φ and the conditional hit distribution. Ablations: zero the f/l **input channels to the residual CNN while retaining the anchor**; separately replace the anchor with uniform logits on occupied rays. These counterfactuals are diagnostics, not proof of voxel attribution.

Its receptive field along y lets a column borrow evidence from its neighbours.

**Fit layer.** Take moments μ and s² of φ. Use a variance floor, a cap, and a strictly positive ridge: `w = min(w_max, π̂/(s² + 0.25)) + ε_ridge`, with w_max and ε_ridge (e.g. 1e-3) fixed before evaluation. The cap stops a narrow but wrong peak from dominating the fit; ε_ridge keeps the solve well conditioned on absent stretches. **Check:** perturb the edge moments on absent rays and measure the change in L̂, Û on neighbouring occupied rays; it should be negligible. Then solve

```
min_h  Σ_y w_y (h_y − μ_y)² + Σ_y λ π̂_y π̂_{y+1} · ρ(h_{y+1} − h_y)
```

The product of predicted presences is a soft connection, not an exact break across disconnected runs. Inspect leakage across true gaps without using the ground-truth run mask in the forward fit.

- quadratic penalty on adjacent differences: a tridiagonal solve. This is the first implementation.
- absolute-difference penalty (1-D TV / fused lasso): a piecewise-constant curve. Its backward pass is not generally just block averaging here, because the weights, presence gates and potential order constraint also depend on the head. Add this arm after a solver-specific gradient check.

This gives fitted curves L̂_r and Û_r. Add `λ_order · mean relu(L̂−Û)` as an initial crossing penalty and report the fraction and size of crossings. A hinge reduces crossings but does **not** guarantee `L̂≤Û`; if crossings persist, use an ordered parameterization or joint constrained fit. A crossed pair tends to produce weak occupancy, which is a segmentation failure rather than a safe solution.

## 2. Curve loss (updates head and voxel network)

Each head's likelihood is unconditional (presence × edge | occupied), and the two NLLs are summed, so the shared presence appears **once per head**:

```
occupied ray:  NLL_r = −2·log π̂_r − log φ^L_r(L*_r) − log φ^U_r(U*_r)
absent ray:    NLL_r = −2·log(1 − π̂_r)

L_edge  = Σ_r ω_r NLL_r / Σ_r ω_r        ω_r = 1 in the dilated GT box; outside it, absent rays are
                                          down-weighted or subsampled (ratio fixed before evaluation)
L_curve = L_edge + mean_{r: q*_r=1} [ |L̂_r − L*_r| + |Û_r − U*_r| ]
L_order = mean_{r} relu(L̂_r − Û_r)
```

At g = 0 an occupied ray's NLL is exactly `−log f(L*) − log l(U*)`, whose gradient is zero on voxels strictly between the edges. **Counting presence only once is wrong:** one head's `+log π` survives and the wrong-sign interior gradient returns (+0.038 on z = 4–6 in the check, vs 0.000 counted twice). An absent ray's NLL at g = 0 is `−2 log m = −2 Σ_t log(1 − p_t)`, which pushes every voxel on it toward background, as it should.

Presence is trained on **all** rays: the renderer acts everywhere at inference, so an untrained π̂ outside the box could add false-positive foreground. Report false-positive components far from the GT hippocampus.

- Start with a one-hot integer-edge target. Target width is an ablation; one worked column does not establish that every wider Gaussian gives an incorrect voxel gradient.
- L1 matches edge MAE and is robust to large errors. The fit regularizer, rather than L1 supervision, determines whether the predicted curve has plateaus. Smooth L1 is a training-stability option.

Because p, f and l are head inputs and are not detached, ∂L_curve/∂a is non-zero. **Caveat:** that gradient is "whatever makes the head right", passed through the head's Jacobian. It is not guaranteed to point toward the correct voxel labels, and nothing stops the voxel net from shaping p to make the head's job easier. That is why §3 exists.

## 3. Responsible-voxel loss (GT-anchored voxel gradient, voxel network only)

### Why it cannot be derived from a curve or distribution distance

Take W₁ on the voxel first-hit distribution, with ground-truth edge L* = 3, a stray voxel at z = 1 (p = 0.6), and a **missing** true edge (p₃…p₅ = 0.05, first real hit at z = 6).

- The W₁ gradient on the stray voxel's logit is **−0.197**, which pushes it **on**: W₁ prefers "2 voxels early" to "3 voxels late".
- The missing edge voxel gets only −0.055.

A pure distribution or curve distance can therefore reward a wrong voxel. The voxel correction must come from the labels.

### Sets per ray, for band half-width b (use 0 or 1)

```
Neg_r = { z : y*_z = 0  and (z < L*_r or z > U*_r) }        # all background outside the true run
Pos_r = { z : y*_z = 1  and (z ≤ L*_r + b or z ≥ U*_r − b) } # GT edge band
```

For absent rays, Neg_r is the whole ray.

### Responsibility weights (stop-gradient)

- On Neg: r_z = sg[max(f_r(z), l_r(z))]. This measures how much the raw voxel prediction is *currently acting as* the first or last hit. Because the residual CNN can use other features, it is a targeted proxy, not a measurement of how much the head used that voxel.
- On Pos: r_z = 1. A missed edge has almost no first-hit mass (f(3) = 0.019 in the example above), so predicted weights cannot find it. That is the reason for the ground-truth band.

### Loss

```
L_resp = mean_{r∈box}  Σ_z r_z · BCEwithLogits(a_z, y*_z)  /  max(1, Σ_z r_z)
```

The **per-voxel logit gradient** is bounded: `r_z(p_z − y*_z)/max(1, Σr)` has magnitude ≤ 1. The loss value itself is not bounded. The max(1, ·) stops a ray holding only a tiny stray from being amplified. Raw three-class Dice+CE remains responsible for background holes within `[L*,U*]` and other unselected voxels.

**Do not clip probabilities to bound the value.** Clipping p to [ε, 1−ε] kills the gradient exactly on the worst-missed edge voxels (p < ε).

A known limitation: a stray voxel shadowed by an earlier stray has small f and is corrected only once the earlier one is gone.

### Numerical check

Gradient on the logits (positive = pushed to background). Central voxels are untouched.

| column | z=1 | z=2 | z=3 | z=4 | z=5 | z=6 |
|---|---|---|---|---|---|---|
| shifted (excess z=2, missing z=6), b=0 | 0 | **+0.263** | −0.017 | 0 | 0 | **−0.304** |
| stray z=1 + missing edge z=3, b=0 | **+0.132** | 0 | **−0.355** | 0 | 0 | −0.019 |
| same, b=1 | +0.075 | 0 | −0.203 | −0.203 | −0.203 | −0.011 |

In the second case W₁ pushes the stray voxel the wrong way; L_resp pushes it the right way.

## 4. Soft rendering into the foreground logits + ordinary 3-class Dice

### Soft occupancy of the predicted interval

```
o_r(z) = π̂_r · σ((z − L̂_r + ½)/τ) · σ((Û_r − z + ½)/τ),    τ ≈ 0.5 voxel (optionally annealed)
```

### Fusion: shift only the background logit

```
ℓ'_B = ℓ_B − β(t) · M·tanh(logit(o)/M),   ℓ'_A = ℓ_A,   ℓ'_P = ℓ_P
β(t) = warmup(t) · β_max · sigmoid(β_raw),   warmup(0) = 0,   M ≈ 6
```

- Since only ℓ_B moves, s'_A/s'_P = exp(ℓ_A − ℓ_P) is unchanged. The function changes *whether* a voxel is hippocampus, never *which* class. (Checked numerically: A/P ratio 2.0138 before and after; foreground 0.476 → 0.832 at o = 0.7, β = 2.)
- Compute log o as a sum of logsigmoids and log(1−o) = log(−expm1(log o)), so π̂ → 0 still passes gradient. Bounding β as well as the occupancy logit makes the maximum shift explicit. Start at β = 0, then ramp it after the head begins to learn. Check sensitivity to τ rather than fixing 0.5 voxel from the outset.
- **Asymmetry and thin rays.** Inside the interval the background shift is at most ≈ β·logit(o) (about 2–3β), while outside it saturates at β·M = 6β, so the renderer suppresses far more readily than it adds. A one-voxel interval (L̂ = Û, common at tail and apex) peaks at σ(½/τ)² ≈ 0.53 occupancy at τ = 0.5 before presence (logit ≈ 0.14), so it adds almost nothing there. Test τ ∈ {0.25, 0.5} and report thin-ray (run length ≤ 2) errors separately. Alternative: an **interval-distance renderer** `o = π̂·σ(((Û − L̂ + 1)/2 − |z − (L̂+Û)/2|)/τ)`, a single sigmoid that gives 0.73 (τ = 0.5) or 0.88 (τ = 0.25) at a one-voxel interval.

### Loss

```
L_seg = [DiceCE_3class(softmax ℓ') + η · DiceCE_3class(softmax ℓ)] / (1 + η)
L = L_seg + λ_c L_curve + λ_r L_resp + λ_order L_order
```

- Dice on ℓ' sends gradient into L̂, Û and π̂ through ∂o/∂L̂ = −o(1 − σ_L)/τ, and from there into the head and, via its inputs, into the voxel net. This is what makes the function consequential.
- The auxiliary η-term (η ≈ 0.5) keeps the voxel network good on its own, so the renderer cannot carry it.
- The 1/(1 + η) normalization keeps the total segmentation weight constant through warm-up: while β = 0, ℓ′ = ℓ and both Dice terms are identical.
- At inference, output softmax(ℓ'), and also report the unrendered ℓ.

**Multi-run caveat.** The renderer favors filling `[L̂, Û]`, including gaps in GT rays with two runs. In the native fold-0 data audit, 410/112,904 occupied training rays (0.363%) and 132/29,019 occupied outer-validation rays (0.455%) have two runs; their internal holes are 584/682,404 (0.0856%) and 192/174,350 (0.1101%) of foreground-voxel counts, respectively. These numbers favor testing full soft rendering while retaining raw Dice+CE, and reporting multi-run errors separately. Compare the *outside-only* variant `ℓ'_B = ℓ_B − β·min(0, logit o)` as an ablation: it suppresses voxels outside the interval, leaving missed edges to the raw network and `L_resp`.

## Gradient routing

| term | voxel network | function head |
|---|---|---|
| L_seg (normalized DiceCE(ℓ′) + η·DiceCE(ℓ)) | direct; and via render → curves → φ, π̂ → {anchor A, residual inputs} | via render → curves |
| L_edge, anchor path (A = log f − log π, log m) | at g = 0: exactly `−log f(L*) − log l(U*)` (+ presence on absent rays); label-aligned at initialization | — |
| L_edge / L1, residual-input path (p, f, l channels of the CNN) | **no sign guarantee** | direct |
| L_resp | direct, per-voxel, GT-anchored | — (weights stop-grad) |

"Label-aligned" holds only at the zero-residual initialization and only for the direct edge NLL. Once residuals, the curve fit, the L1 terms and rendered Dice contribute, the total anchor-path gradient has **no general sign guarantee**. Measure each path separately (detach the anchor, then the input channels): report gradient norms into the raw voxel logits and their cosine with the L_resp gradient on the GT edge band.

Inspect gradient norms on training cases when selecting λ_c and λ_r; the suggested 0.1–0.3× ratio is a tuning hypothesis, not a fixed requirement. Ramp bounded β from exactly zero after the head warms up.

## Implementation gates (before any full run)

- **Patch borders:** mask rays whose GT or predicted run touches a z-border of the training patch; a 64³ default tensor does not by itself prove rays are untruncated.
- **Precision:** run the log-space hit computation, the log π/`expm1` step and the solve in float32 under AMP.
- **Resolution:** apply the new terms at full resolution only (not to deep-supervision outputs).
- **Exact tie:** assert `P^L = f`, `P^U = l`, `P(∅) = m` at initialization to float tolerance.
- **Gradient signs:** on synthetic columns, check the L_edge and L_resp gradient signs listed above.



```python
# logits: B×3×X×Y×Z (bg, ant, post); y3: B×X×Y×Z in {0,1,2}; z is the ray axis
a  = torch.logsumexp(logits[:, 1:], 1) - logits[:, 0]           # fg logit, B×X×Y×Z
yf = y3 > 0
Z  = a.shape[-1]; z = torch.arange(Z, device=a.device)

def hit_dists(a):
    lp, lq = F.logsigmoid(a), F.logsigmoid(-a)
    cq = torch.cumsum(lq, -1)
    log_f = lp + F.pad(cq[..., :-1], (1, 0))
    rq = torch.flip(torch.cumsum(torch.flip(lq, [-1]), -1), [-1])
    log_l = lp + F.pad(rq[..., 1:], (0, 1))
    return log_f.exp(), log_l.exp(), log_f, log_l

f, l, log_f, log_l = hit_dists(a)                               # NOT detached

present = yf.any(-1)
Ls = torch.where(present, yf.float().argmax(-1), torch.full_like(present, Z, dtype=torch.long))
Us = torch.where(present, Z - 1 - yf.flip(-1).float().argmax(-1), torch.full_like(Ls, -1))

# --- §3 responsibility loss
b = 1
neg = (~yf) & ((z < Ls[..., None]) | (z > Us[..., None]))
pos = yf & ((z <= Ls[..., None] + b) | (z >= Us[..., None] - b))
r = torch.where(neg, torch.maximum(f, l).detach(), torch.zeros_like(f)) + pos.float()
bce = F.binary_cross_entropy_with_logits(a, yf.float(), reduction='none')
L_resp = ((r * bce).sum(-1) / r.sum(-1).clamp_min(1.0))[in_box].mean()

# --- §1-2 head (fold X into batch; each sagittal slice is a (Y,Z) image)
B, X, Y, _ = a.shape
log_m  = torch.cumsum(F.logsigmoid(-a), -1)[..., -1]           # log miss, B×X×Y  (float32)
log_pi = torch.log(-torch.expm1(log_m.clamp(max=-1e-7)))      # log voxel presence
inp = torch.cat([a.sigmoid()[:, None], f[:, None], l[:, None], proj(feats)], 1)
inp2d = inp.permute(0, 2, 1, 3, 4).reshape(B * X, -1, Y, Z)
gL, gU, g_pi = head2d(inp2d)          # (B*X,Y,Z), (B*X,Y,Z), (B*X,Y); final layers zero-init
AL = (log_f - log_pi[..., None]).reshape(B * X, Y, Z)          # conditional log first-hit mass
AU = (log_l - log_pi[..., None]).reshape(B * X, Y, Z)
logphiL = (AL + gL).log_softmax(-1)
logphiU = (AU + gU).log_softmax(-1)
logit_pres = (log_pi - log_m).reshape(B * X, Y) + g_pi        # ONE presence for both heads
log_pres, log_abs = F.logsigmoid(logit_pres), F.logsigmoid(-logit_pres)

occ = present.reshape(B * X, Y)
iL = Ls.reshape(B * X, Y, 1).clamp(max=Z - 1)
iU = Us.reshape(B * X, Y, 1).clamp(min=0)
nll = torch.where(occ,
                  -2 * log_pres - logphiL.gather(-1, iL)[..., 0] - logphiU.gather(-1, iU)[..., 0],
                  -2 * log_abs)                                # presence counted once per head
w_ray = ray_weight.reshape(B * X, Y)                           # 1 in box; reweighted outside
L_edge = (nll * w_ray).sum() / w_ray.sum()

pres = log_pres.exp()
Lhat, Uhat = fit(logphiL.exp(), pres), fit(logphiU.exp(), pres) # quadratic first; w floor = ridge
L_order = (Lhat - Uhat).relu().mean()                           # penalty, not a guarantee

# --- §4 soft render into the background logit only
zz = z.float()
log_o = (log_pres[..., None]
         + F.logsigmoid((zz - Lhat[..., None] + .5) / tau)
         + F.logsigmoid((Uhat[..., None] - zz + .5) / tau))
logit_o = log_o - torch.log(-torch.expm1(log_o.clamp(max=-1e-6)))
shift = (M * torch.tanh(logit_o / M)).reshape(B, X, Y, Z)
beta = warmup * beta_max * torch.sigmoid(beta_raw)              # warmup=0 initially
logits_r = torch.cat([(logits[:, 0] - beta * shift)[:, None], logits[:, 1:]], 1)
```

## Experiment arms (add to the existing plan)

1. A: matched three-class Dice+CE baseline.
2. B: A + `L_resp` only. This isolates the GT-anchored voxel term; it is close to boundary-banded BCE.
3. C: A + residual miss-bin head + `L_curve`, with β = 0. This isolates head supervision without rendering.
4. D: B + residual miss-bin head + `L_curve`, with β = 0.
5. **D′: D rendered at inference only.** Same trained weights as D; β and τ chosen by the same selection rule and data as E. E vs D′ is the test of the central claim, that training **through** the renderer helps beyond test-time rendering.
6. E: D + bounded, warmed-up full renderer and quadratic fit.
7. F: E with a differentiable TV fit, after its backward pass passes gradient checks; compare the outside-only and interval-distance renderers separately.

**Primary endpoint:** case-level foreground-union average symmetric surface distance (ASSD, in physical millimeters) on the final rendered segmentation, assessed on a split locked before the confirmatory run. Report HD95, union and class Dice, presence error, raw and final contour MAE, voxels fixed/broken by rendering, crossing rate, and multi-run-ray error. Do not use the already explored outer validation for a confirmatory claim. Use matched data, training budget, and model selection rules across arms.

**Next implementation slice:** (1) measure the log-hit gap distribution from a baseline checkpoint (§1 rescue gate); (2) build B (`L_resp` only) and the residual miss-bin head with β fixed at 0; (3) pass the implementation gates (exact tie, gradient signs, patch borders, float32). Then verify edge targets, curve learning, crossing rate, and presence calibration inside **and outside** the GT box before adding rendering.
