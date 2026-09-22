# Function head with voxel feedback: loss derivation (22 Sep 2026)

Status: design specification, not an implemented or validated model. The current fold-0 outer validation has informed the design and is exploratory; confirmatory evaluation needs a locked, previously untouched split or nested cross-validation.

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

Rays enter the losses only inside a dilated slice bounding box, so that thousands of trivially empty rays do not dominate.

## 1. Differentiable first- and last-hit distributions (voxel path, not detached)

The foreground logit is a_z = logsumexp(ℓ_A, ℓ_P) − ℓ_B, with p_z = σ(a_z).

```
first hit   f_r(z) = p_z · Π_{t<z} (1 − p_t)
last hit    l_r(z) = p_z · Π_{t>z} (1 − p_t)
miss        m_r    = Π_t (1 − p_t),     voxel presence π_r = 1 − m_r
```

Compute everything in log space: log p = logsigmoid(a), log(1−p) = logsigmoid(−a), then an exclusive cumsum (a reverse one for l). This is O(Z), runs on the GPU and has no Python loop.

**Head input.** Treat each sagittal slice as a 2-D (y, z) image with channels [p, f, l, projected decoder features]. A small 2-D CNN predicts residual edge logits g^L and g^U, plus a separate per-ray presence logit ρ. Use the actual first/last-hit log masses as the default edge logits:

```
e^L(y,z) = log f(y,z) + g^L(y,z)
e^U(y,z) = log l(y,z) + g^U(y,z)
φ^L = softmax_z(e^L),  φ^U = softmax_z(e^U)
```

Calculate log f and log l with the log-space hit code, not by taking log of rounded probabilities. Zero-initialize the final residual layer. On an occupied ray, the initial softmax is then exactly the **first/last-hit distribution conditional on at least one hit**; f and l themselves sum to voxel presence π, not to one. For a nearly empty ray, the conditional distribution can be arbitrary, so the presence branch and presence loss remain essential. Image features let g rescue a true edge with very low raw hit mass. Keep a logit floor only for finite precision; avoid a bounded g that would make rescue impossible.

Report the centered residual magnitude `mean_z |g(z) − mean_z g|` on occupied rays (a constant logit offset has no effect on softmax), and the distance between φ and the conditional hit distribution. For a diagnostic ablation, zero the f/l **input channels to the residual CNN while retaining the log-hit anchor**; separately replace the anchor with uniform logits on occupied rays. Zeroing the anchor itself would make `log f` invalid and conflate two paths. These counterfactuals are diagnostics, not proof of exact voxel attribution. Inspect the actual `L_curve` gradient into raw voxel logits as well.

The head produces:

- residual edge logits g^L(y,z) and g^U(y,z), added to log f and log l before softmax → head distributions φ^L_r and φ^U_r;
- a separately pooled presence logit ρ_r. Do not assume max pooling alone is calibrated across 64 opportunities.

Its receptive field along y lets a column borrow evidence from its neighbours.

**Fit layer.** Take moments μ and s² of φ. Use a variance floor and bounded weights, for example `w = clamp(σ(ρ)/(s² + 0.25), min=w_min, max=w_max)`, with w_min and w_max selected before evaluation. A narrow but wrong peak should not dominate the fit. Then solve

```
min_h  Σ_y w_y (h_y − μ_y)² + Σ_y λ σ(ρ_y) σ(ρ_{y+1}) · ρ(h_{y+1} − h_y)
```

The product of predicted presences is a soft connection, not an exact break across disconnected runs. Inspect leakage across true gaps without using the ground-truth run mask in the forward fit.

- quadratic penalty on adjacent differences: a tridiagonal solve. This is the first implementation.
- absolute-difference penalty (1-D TV / fused lasso): a piecewise-constant curve. Its backward pass is not generally just block averaging here, because the weights, presence gates and potential order constraint also depend on the head. Add this arm after a solver-specific gradient check.

This gives fitted curves L̂_r and Û_r. Add `λ_order · mean relu(L̂−Û)` as an initial crossing penalty and report the fraction and size of crossings. A hinge reduces crossings but does **not** guarantee `L̂≤Û`; if crossings persist, use an ordered parameterization or joint constrained fit. A crossed pair tends to produce weak occupancy, which is a segmentation failure rather than a safe solution.

## 2. Curve loss (updates head and voxel network)

```
L_curve = mean_{r∈box} BCE(ρ_r, q*_r)
        + mean_{r: q*_r=1} [ CE(φ^L_r, δ_{L*}) + CE(φ^U_r, δ_{U*}) + |L̂_r − L*_r| + |Û_r − U*_r| ]
L_order = mean_{r∈box} relu(L̂_r − Û_r)
```

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
o_r(z) = σ(ρ_r) · σ((z − L̂_r + ½)/τ) · σ((Û_r − z + ½)/τ),    τ ≈ 0.5 voxel (optionally annealed)
```

### Fusion: shift only the background logit

```
ℓ'_B = ℓ_B − β(t) · M·tanh(logit(o)/M),   ℓ'_A = ℓ_A,   ℓ'_P = ℓ_P
β(t) = warmup(t) · β_max · sigmoid(β_raw),   warmup(0) = 0,   M ≈ 6
```

- Since only ℓ_B moves, s'_A/s'_P = exp(ℓ_A − ℓ_P) is unchanged. The function changes *whether* a voxel is hippocampus, never *which* class. (Checked numerically: A/P ratio 2.0138 before and after; foreground 0.476 → 0.832 at o = 0.7, β = 2.)
- Compute log o as a sum of logsigmoids and log(1−o) = log(−expm1(log o)), so σ(ρ) → 0 still passes gradient. Bounding β as well as the occupancy logit makes the maximum shift explicit. Start at β = 0, then ramp it after the head begins to learn. Check sensitivity to τ rather than fixing 0.5 voxel from the outset.

### Loss

```
L = DiceCE_3class(softmax ℓ') + η · DiceCE_3class(softmax ℓ)
    + λ_c L_curve + λ_r L_resp + λ_order L_order
```

- Dice on ℓ' sends gradient into L̂, Û and ρ through ∂o/∂L̂ = −o(1 − σ_L)/τ, and from there into the head and, via its inputs, into the voxel net. This is what makes the function consequential.
- The auxiliary η-term (η ≈ 0.5) keeps the voxel network good on its own, so the renderer cannot carry it.
- At inference, output softmax(ℓ'), and also report the unrendered ℓ.

**Multi-run caveat.** The renderer favors filling `[L̂, Û]`, including gaps in GT rays with two runs. In the native fold-0 data audit, 410/112,904 occupied training rays (0.363%) and 132/29,019 occupied outer-validation rays (0.455%) have two runs; their internal holes are 584/682,404 (0.0856%) and 192/174,350 (0.1101%) of foreground-voxel counts, respectively. These numbers favor testing full soft rendering while retaining raw Dice+CE, and reporting multi-run errors separately. Compare the *outside-only* variant `ℓ'_B = ℓ_B − β·min(0, logit o)` as an ablation: it suppresses voxels outside the interval, leaving missed edges to the raw network and `L_resp`.

## Gradient routing

| term | voxel network | function head |
|---|---|---|
| DiceCE(ℓ') | direct, and via render → curves → head inputs | via render → curves |
| η·DiceCE(ℓ) | direct | — |
| L_curve | via head inputs (p, f, l): direction not controlled | direct |
| L_resp | direct, per-voxel, GT-anchored | — (weights stop-grad) |

Inspect gradient norms on training cases when selecting λ_c and λ_r; the suggested 0.1–0.3× ratio is a tuning hypothesis, not a fixed requirement. Ramp bounded β from exactly zero after the head warms up. Measure the sign of the responsible-voxel gradient and the magnitude of the curve-loss gradient into the decoder before a full run.

## Reference implementation sketch (PyTorch)

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
inp = torch.cat([a.sigmoid()[:, None], f[:, None], l[:, None], proj(feats)], 1)
inp2d = inp.permute(0, 2, 1, 3, 4).reshape(B * X, -1, Y, Z)
h = head2d(inp2d)                                              # 2 residual edge maps
gL, gU = h[:, 0], h[:, 1]                                     # zero-initialize their last layer
log_f2d = log_f.reshape(B * X, Y, Z)
log_l2d = log_l.reshape(B * X, Y, Z)
phiL = (log_f2d + gL).softmax(-1)
phiU = (log_l2d + gU).softmax(-1)
rho = presence_head(inp2d)                                     # separately pooled, (B*X, Y)
# fit computes moment/variance and bounded weights, then solves along Y.
Lhat, Uhat = fit(phiL, rho.sigmoid()), fit(phiU, rho.sigmoid()) # quadratic first
L_order = (Lhat - Uhat).relu().mean()                           # penalty, not a guarantee

# --- §4 soft render into the background logit only
zz = z.float()
log_o = (F.logsigmoid(rho)[..., None]
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
3. C: A + residual head + `L_curve`, with β = 0. This isolates head supervision without rendering.
4. D: B + residual head + `L_curve`, with β = 0. This is the direct control for rendering in E.
5. E: D + bounded, warmed-up full renderer and quadratic fit.
6. F: E with a differentiable TV fit, after its backward pass passes gradient checks; compare the outside-only renderer separately.

**Primary endpoint:** case-level foreground-union average symmetric surface distance (ASSD, in physical millimeters) on the final rendered segmentation, assessed on a split locked before the confirmatory run. Report HD95, union and class Dice, presence error, raw and final contour MAE, voxels fixed/broken by rendering, crossing rate, and multi-run-ray error. Do not use the already explored outer validation for a confirmatory claim. Use matched data, training budget, and model selection rules across arms.

**Next implementation slice:** build B (`L_resp` only) and the residual head with β fixed at 0. Verify edge targets, raw-mask gradients, curve learning, crossing rate, and presence calibration before adding rendering.
