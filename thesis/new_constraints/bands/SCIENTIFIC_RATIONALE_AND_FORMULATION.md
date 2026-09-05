# Outer-boundary band constraint: rationale and mathematical formulation

## Purpose

The outer-boundary band constraint is a deliberately narrow auxiliary loss. It
does not try to solve the internal anterior/posterior separation. It asks a
simpler question near the ground-truth (GT) hippocampus contour:

- are voxels immediately inside the contour predicted as hippocampus?
- are voxels immediately outside the contour predicted as background?

This is a sensible first constraint because the fold-0 error analysis shows that
the dominant remaining failure is not a distant false component or an
anterior/posterior swap. It is a foreground/background decision displaced by a
small number of voxels around the outer contour.

The canonical implementation is in `outer_boundary.py`. Weight calibration is
in `calibrate_weight.py`, and integration with the common `none`, `equivariance`,
and `bands` pipeline is in `../objective.py` and
`../train_swinunetr_constraints.py`.

## 1. Empirical reason for the constraint

### 1.1 Evidence used to define the hypothesis

The analysis used the same 52 held-out patients in MSD fold 0, with predictions
in the 64 x 64 x 64 model space. The baseline checkpoint was the matched
`constraint-set none` run:

```text
msd_fold0_none_20260806_110334_616958/checkpoint_best.pt
```

The comparison model was the matched translation-equivariance run:

```text
msd_fold0_translation_20260805_210911_616615/checkpoint_best.pt
```

Each wrong voxel was assigned exactly one of six directed error types:

1. background -> anterior;
2. background -> posterior;
3. anterior -> background;
4. posterior -> background;
5. anterior -> posterior;
6. posterior -> anterior.

The first two are foreground false positives (FP), the next two are foreground
false negatives (FN), and the last two are anterior/posterior swaps.

### 1.2 Baseline error counts

The baseline produced 39,781 wrongly labelled voxels, or 765.0 per patient on
average.

| Error group | Voxels | Share of all errors |
|---|---:|---:|
| Foreground false positives | 19,512 | 49.0% |
| Foreground false negatives | 16,397 | 41.2% |
| Anterior/posterior swaps | 3,872 | 9.7% |
| **All foreground/background errors** | **35,909** | **90.3%** |

The directed counts show that this is not a single-class artifact:

| GT -> prediction | Voxels | Share of all errors |
|---|---:|---:|
| Background -> anterior | 10,176 | 25.6% |
| Background -> posterior | 9,336 | 23.5% |
| Anterior -> background | 7,833 | 19.7% |
| Posterior -> background | 8,564 | 21.5% |
| Anterior -> posterior | 1,486 | 3.7% |
| Posterior -> anterior | 2,386 | 6.0% |

The spatial localization is even more informative:

- 36,372 of 39,781 errors, **91.4%**, are within one voxel unit of the GT
  outer boundary;
- 39,159 of 39,781 errors, **98.4%**, are within two voxel units;
- only 622 errors, **1.6%**, lie farther than two voxel units from that boundary.

These distances were measured with a Euclidean distance transform in voxel
coordinates. They include all error types, including anterior/posterior swaps.
They therefore motivate, but do not mathematically equal, the 6-connected
morphological bands defined later.

![Baseline boundary proximity](../../../evaluation/fold0_voxel_error_results/plots/03_boundary_proximity.png)

The errors also have structured slice locations in the 64-cubed model space.
The "central 80%" columns give the smallest reported slice interval containing
80% of the corresponding aggregate error mass across the 52 patients.

| Axis | Peak all-error slice | Central 80% of all errors | Peak FP | Peak FN | Peak A/P swap | Central 80% of swaps |
|---|---:|---:|---:|---:|---:|---:|
| Sagittal | 32 | 22-38 | 32 | 34 | 25 | 21-31 |
| Coronal/AP | 34 | 16-47 | 45 | 16 | 34 | 32-36 |
| Axial | 30 | 22-38 | 29 | 31 | 27 | 24-31 |

The sharpest internal pattern is along the coronal/AP axis, where A/P swaps
concentrate near the class cutoff. The outer FP and FN errors occupy broader and
different slice ranges. The band constraint therefore does not hard-code a
global slice index. It follows each patient's transformed GT contour wherever
that contour lies.

### 1.3 What translation equivariance changes, and what it leaves unresolved

Translation equivariance improves the same 52 patients:

| Quantity | Baseline | Translation equivariance | Change |
|---|---:|---:|---:|
| Mean hard Dice | 0.874527 | 0.882488 | +0.007961 |
| Total wrong voxels | 39,781 | 37,319 | -2,462 (-6.19%) |
| Wrong voxels per patient | 765.02 | 717.67 | -47.35 |
| Foreground FP | 19,512 | 18,006 | -1,506 |
| Foreground FN | 16,397 | 15,782 | -615 |
| Anterior/posterior swaps | 3,872 | 3,531 | -341 |

The paired voxel comparison also shows that translation equivariance fixes
7,532 baseline errors but introduces 5,070 errors at previously correct voxels.
There are 32,037 voxels that retain the same wrong label. The constraint helps,
but it is not direct supervision of the contour.

The candidate's remaining errors have essentially the same localization:

- 92.6% are within one voxel unit of the GT outer boundary;
- 98.8% are within two voxel units;
- foreground FP and FN still account for 33,788 of 37,319 errors, or 90.5%.

This gives the band experiment a precise role. Translation equivariance asks the
network to respond consistently when the input is shifted. The band loss asks
the network to put the outer foreground/background transition in the correct
place. These are different objectives.

### 1.4 The conclusion supported by the data

The evidence supports the following testable hypothesis:

> Giving balanced, direct foreground/background supervision in a two-step band
> around the GT outer contour should reduce the dominant FP and FN errors without
> changing the definition of the anterior/posterior task.

It does not prove that two morphological steps are optimal. The radius and this
hypothesis were chosen after inspecting fold 0, so a fold-0 bands result is
exploratory. A later result on an untouched fold is needed for independent
confirmation.

## 2. Mathematical definition of the bands

### 2.1 Labels and foreground union

Let the voxel grid be

$$
\Omega = \{1,\ldots,X\}\times\{1,\ldots,Y\}\times\{1,\ldots,Z\}.
$$

For the three-class MSD model, the GT label at voxel $x\in\Omega$ is

$$
g(x)\in\{0,A,P\},
$$

where $0$ is background, $A$ is anterior hippocampus, and $P$ is
posterior hippocampus. The outer-contour experiment merges the two hippocampal
classes into one GT foreground set:

$$
H = \{x\in\Omega: g(x)\in\{A,P\}\}.
$$

Equivalently, its binary target is

$$
y_H(x)=\mathbf 1[x\in H].
$$

This union is intentional. The loss has no anterior-versus-posterior target.

### 2.2 Six-connected neighbourhood

For a voxel $x=(i,j,k)$, define the seven-position cross

$$
N_6(x)=\{x,(i-1,j,k),(i+1,j,k),(i,j-1,k),(i,j+1,k),(i,j,k-1),(i,j,k+1)\}
$$

Only face-sharing neighbours count. Diagonal neighbours do not. The code
implements this set with a 3 x 3 x 3 cross-shaped convolution kernel containing
seven ones: one at the centre and one in each axial direction. Extend the set
indicator outside the image by zero:

$$
\widetilde{\mathbf 1}_S(u)=
\begin{cases}
1,&u\in S,\\
0,&u\notin S,
\end{cases}
$$

where the second case includes coordinates outside $\Omega$.

For a binary set $S\subseteq\Omega$, one 6-connected dilation and erosion are

$$
D_1(S)=\{x\in\Omega:\exists u\in N_6(x),\;
\widetilde{\mathbf 1}_S(u)=1\},
$$

$$
E_1(S)=\{x\in\Omega:\forall u\in N_6(x),\;
\widetilde{\mathbf 1}_S(u)=1\}.
$$

The two-step operations are recursive:

$$
D_2(S)=D_1(D_1(S)),\qquad E_2(S)=E_1(E_1(S)).
$$

Zero padding makes locations outside the image behave as background during the
convolutions. A foreground object touching an image edge can therefore have an
unusual or truncated band, which is why edge-touching patients are logged.

### 2.3 Inner and outer bands

The canonical bands are

$$
B_{\mathrm{in}} = H\setminus E_2(H),
$$

$$
B_{\mathrm{out}} = D_2(H)\setminus H.
$$

The set subtraction explains their GT class membership:

- $B_{\mathrm{in}}\subseteq H$, so every inner-band voxel is GT foreground;
- $B_{\mathrm{out}}\subseteq\Omega\setminus H$, so every outer-band voxel is
  GT background;
- $B_{\mathrm{in}}\cap B_{\mathrm{out}}=\varnothing$.

The loss is not applied to every voxel. It is evaluated only where either band
mask is true. Voxels in the deep foreground and distant background receive no
band-loss gradient, although they still receive the ordinary Dice gradient.

Two iterations define a two-step Manhattan or morphological distance. For
example, an offset $(1,1,0)$ requires two face-sharing moves and is reached,
while $(1,1,1)$ requires three and is not. This differs from a two-voxel
Euclidean sphere and from a physical 2 mm band.

The bands are constructed from the final transformed label passed to the loss.
In this pipeline that means after the configured resize, or after spatial pad
and centre crop when resize is disabled. Image and label undergo the same spatial
operation, and nearest-neighbour interpolation is used for labels.

## 3. Turning three-class logits into foreground log-odds

### 3.1 General grouped form

Let $z_c(x)$ be the model logit for class $c$ at voxel $x$. Let $F$ be
the foreground-class IDs and $C$ their complement. The implementation requires
$F$ and $C$ to be nonempty, disjoint, and together partition every output
class.

The grouped foreground-versus-complement log-odds are

$$
r_H(x)
=
\operatorname{logsumexp}_{c\in F}z_c(x)
-
\operatorname{logsumexp}_{c\in C}z_c(x).
$$

For MSD, $F=\{A,P\}$ and $C=\{0\}$, so

$$
r_H(x)=\operatorname{logsumexp}(z_A(x),z_P(x))-z_0(x).
$$

This is not an approximation. It is exactly the foreground-versus-background
log-odds induced by the three-class softmax.

### 3.2 Proof that the sigmoid is the summed foreground probability

Define

$$
S_F=\sum_{c\in F}e^{z_c},\qquad S_C=\sum_{c\in C}e^{z_c}.
$$

Then

$$
r_H=\log S_F-\log S_C=\log\left(\frac{S_F}{S_C}\right).
$$

Applying the sigmoid gives

$$
\begin{aligned}
\sigma(r_H)
&=\frac{1}{1+e^{-r_H}}\\
&=\frac{1}{1+S_C/S_F}\\
&=\frac{S_F}{S_F+S_C}\\
&=\sum_{c\in F}\frac{e^{z_c}}{\sum_j e^{z_j}}\\
&=\sum_{c\in F}p_c.
\end{aligned}
$$

Therefore, in the MSD case,

$$
\sigma(r_H)=p_A+p_P=p_H.
$$

The implementation uses `logsumexp` instead of first computing probabilities
because it remains stable for very large positive or negative logits.

## 4. Band loss

### 4.1 Stable voxelwise binary cross-entropy

For each voxel, the auxiliary loss uses binary cross-entropy with logits:

$$
\ell(x)
=
\operatorname{BCEWithLogits}(r_H(x),y_H(x)).
$$

An equivalent numerically stable expression is

$$
\ell(r,y)=\max(r,0)-ry+\log(1+e^{-|r|}).
$$

Its derivative with respect to the grouped log-odds is

$$
\frac{\partial\ell}{\partial r_H}=\sigma(r_H)-y_H.
$$

This gives the required direction on both sides:

- on $B_{\mathrm{in}}$, $y_H=1$, so the derivative is negative unless the
  foreground probability is already one. Gradient descent increases $r_H$ and
  therefore increases $p_A+p_P$;
- on $B_{\mathrm{out}}$, $y_H=0$, so the derivative is positive unless the
  foreground probability is already zero. Gradient descent decreases $r_H$ and
  therefore decreases $p_A+p_P$.

The loss acts on soft logits during training. The hard dilation and erosion are
applied only to the fixed GT label and are created under `no_grad`; they do not
need to be differentiable.

### 4.2 Equal weighting of the two sides

For patient $n$, let $b_{\mathrm{in}}^{(n)}(x)$ and
$b_{\mathrm{out}}^{(n)}(x)$ be the binary band masks. The implementation first
computes a mean for each side:

$$
L_{\mathrm{in}}^{(n)}
=
\frac{\sum_{x\in\Omega}b_{\mathrm{in}}^{(n)}(x)\ell^{(n)}(x)}
{\sum_{x\in\Omega}b_{\mathrm{in}}^{(n)}(x)+\varepsilon},
$$

$$
L_{\mathrm{out}}^{(n)}
=
\frac{\sum_{x\in\Omega}b_{\mathrm{out}}^{(n)}(x)\ell^{(n)}(x)}
{\sum_{x\in\Omega}b_{\mathrm{out}}^{(n)}(x)+\varepsilon},
$$

with $\varepsilon=10^{-6}$. It then balances the two sides:

$$
L_{\mathrm{band}}^{(n)}
=
\frac{1}{2}L_{\mathrm{in}}^{(n)}
+
\frac{1}{2}L_{\mathrm{out}}^{(n)}.
$$

This two-stage average matters. The outer band often contains more voxels than
the inner band. A single mean over their union would let the larger side dominate
the auxiliary objective. Separate means make a one-unit increase in average
inner error as important as the same increase in average outer error. In practical
terms, contour expansion and contour contraction receive equal nominal weight.

### 4.3 Valid-patient and batch reductions

A patient is valid only when both band counts are nonzero:

$$
v_n=
\mathbf 1[|B_{\mathrm{in}}^{(n)}|>0]
\mathbf 1[|B_{\mathrm{out}}^{(n)}|>0].
$$

The batch loss averages only valid patients:

$$
L_{\mathrm{band}}
=
\frac{\sum_n v_n L_{\mathrm{band}}^{(n)}}{\sum_n v_n}.
$$

Invalid patients are skipped instead of being assigned a zero loss, because a
zero would silently dilute the batch average. Cases can be invalid when the GT
foreground is empty, fills the complete image, or is severely truncated.

If the whole batch is invalid, the implementation returns a finite, float32,
graph-connected zero:

```python
logits.float().reshape(-1)[0] * 0.0
```

This permits a normal backward pass without claiming that the invalid samples
have perfect quality.

## 5. Complete training objective

The existing supervised segmentation loss remains unchanged. For a bands run,
the objective at epoch $e$ is

$$
L_{\mathrm{total}}^{(e)}
=
L_{\mathrm{Dice}}
+
s(e)\lambda_{\mathrm{band}}L_{\mathrm{band}},
$$

where the default five-epoch linear warm-up is

$$
s(e)=\min\left(1,\frac{e}{5}\right).
$$

Thus the calibrated band weight is multiplied by 0.2, 0.4, 0.6, 0.8, and 1.0
in epochs 1 through 5, and by 1.0 afterward. There are no five Dice-only warm-up
epochs.

The presets are mutually exclusive in the current experiment:

- `none`: $\lambda_{\mathrm{band}}=\lambda_{\mathrm{equivariance}}=0$;
- `equivariance`: only the translation-equivariance term is active;
- `bands`: only the outer-boundary band term is active.

The current objective rejects a run in which both auxiliary weights are positive.
This keeps the first comparison interpretable.

## 6. Choosing the band weight from gradients

The band weight is not selected from fold-0 validation performance. It is
calibrated from at most 32 deterministic training cases using an epoch-5,
Dice-only `none` checkpoint trained from scratch with matching data, fold,
transforms, AMP setting, runtime, and source provenance.

For each valid calibration patient $n$, the model forward is computed and the
logits are detached. The script measures gradients with respect to those logits,
not model parameters:

$$
G_D^{(n)}
=
\sqrt{\frac{1}{M}\sum_{j=1}^{M}
\left(\frac{\partial L_{\mathrm{Dice}}^{(n)}}{\partial z_j}\right)^2},
$$

$$
G_B^{(n)}
=
\sqrt{\frac{1}{M}\sum_{j=1}^{M}
\left(\frac{\partial L_{\mathrm{band}}^{(n)}}{\partial z_j}\right)^2},
$$

where $M=CXYZ$. These are unconditional RMS values: locations at which a loss
has zero gradient remain in the denominator. A conditional band-only RMS is also
reported for diagnosis but is not used to select the weight.

Let

$$
D_{50}=\operatorname{median}_n G_D^{(n)},\qquad
B_{50}=\operatorname{median}_n G_B^{(n)},\qquad
B_{95}=Q_{0.95,n}(G_B^{(n)}).
$$

The target weight and safety cap are

$$
\lambda_{\mathrm{target}}
=
0.10\frac{D_{50}}{B_{50}},
$$

$$
\lambda_{\mathrm{cap}}
=
0.50\frac{D_{50}}{B_{95}},
$$

and the training weight is

$$
\boxed{
\lambda_{\mathrm{band}}
=
\min(\lambda_{\mathrm{target}},\lambda_{\mathrm{cap}})
}.
$$

The first expression targets a median unweighted band gradient equal to 10% of
the median Dice gradient after weighting. The second prevents the 95th-percentile
band gradient from exceeding 50% of the median Dice gradient after weighting.
Medians and a high quantile reduce sensitivity to a single unusual patient.

Nonfinite or zero robust gradient magnitudes make calibration fail. An all-invalid
calibration also fails and writes a diagnostic report instead of inventing a
weight.

## 7. Numerical and pipeline implementation contract

The code enforces the following behavior:

1. Labels may have shape `[B, 1, X, Y, Z]` or `[B, X, Y, Z]`; batch and spatial
   dimensions must match the logits.
2. The public canonical loss accepts exactly `steps=2`.
3. The GT masks, grouped `logsumexp`, BCE, reductions, and connected zero are
   computed in float32 even under automatic mixed precision (AMP).
4. Large logits remain finite because grouped probabilities are represented with
   `logsumexp` and the loss uses `binary_cross_entropy_with_logits`.
5. Training computes the model logits once. The same logits feed Dice and the
   band objective.
6. Validation also reuses one base forward per batch for segmentation and band
   diagnostics.
7. Training and validation log raw band loss, inner and outer losses, band voxel
   counts, valid and skipped patients, and edge-touching foregrounds.
8. `exp(-loss)` is retained as a compatibility score for the shared constraint
   interface. It is not the primary scientific outcome and should not be
   interpreted as a probability that the anatomy is correct.

## 8. What the constraint can and cannot fix

### Directly targeted

- foreground predicted just outside the GT contour, corresponding to local FP;
- background predicted just inside the GT contour, corresponding to local FN;
- smooth or displaced outer contours whose errors lie in the selected band.

### Not directly targeted

- anterior predicted as posterior, or posterior predicted as anterior;
- errors more than two 6-connected steps from the GT contour;
- disconnected components or global topology;
- translation consistency;
- physical-distance errors measured in millimetres.

The loss is invariant to swapping the anterior and posterior logit channels. It
may still change their individual gradients because their logits jointly form the
foreground probability, but it contains no target that prefers one internal class
over the other. Anterior Dice, posterior Dice, and A/P swap counts are therefore
required guardrails.

## 9. Evaluation criterion

The primary measure for this experiment is the per-patient hard foreground
symmetric-difference count:

$$
E_H
=
|\widehat H\triangle H|
=
\mathrm{FP}_H+\mathrm{FN}_H.
$$

Lower is better. FP and FN must also be reported separately, because the same
total can hide improvement on one side and deterioration on the other. Macro
anterior/posterior Dice and A/P swaps are guardrails against damage to the
internal labels.

A one-seed run is an engineering pilot. The evidentiary comparison uses matched
seeds and matched patients. With three seeds, the predeclared two-way bootstrap
resamples seeds and patients independently while preserving the arm pairing. The
result remains replicated evidence, not definitive statistical proof.

## 10. Generalization limits

The grouped-logit formulation is dataset-independent when the foreground and
complement class IDs form a complete partition. The present experimental choices
are not automatically dataset-independent:

- class IDs `(1, 2)` versus `(0,)` reflect the current three-class task;
- exactly two steps were motivated by the observed MSD fold-0 error distribution;
- morphological voxel distance depends on the processed grid and ignores physical
  voxel spacing;
- very thin structures can lose most or all of their interior after two erosions.

For a new dataset, the class partition, error-distance distribution, voxel
spacing, and valid-band frequency should be checked before reusing the two-step
setting.

## 11. Evidence and code references

- [Baseline aggregate analysis](../../../evaluation/fold0_voxel_error_results/README.md)
- [Baseline machine-readable summary](../../../evaluation/fold0_voxel_error_results/summary.json)
- [Translation-equivariance aggregate analysis](../../../evaluation/fold0_voxel_error_results_translation/README.md)
- [Exact paired comparison](../../../evaluation/fold0_voxel_error_comparison_translation_vs_none/comparison.json)
- [Band construction and loss](outer_boundary.py)
- [Gradient calibration](calibrate_weight.py)
- [Tests of geometry, gradients, numerical behavior, calibration, and integration](test_outer_boundary.py)
- [Shared constraint objective](../objective.py)
- [Training and validation pipeline](../train_swinunetr_constraints.py)
