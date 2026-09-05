# Focal-logLTN interpretation of the two-band boundary constraint

## Purpose and final classification

This document gives the precise Logic Tensor Network (LTN) interpretation of
the two-band hippocampus boundary constraint implemented in
[`outer_boundary.py`](outer_boundary.py). It separates:

1. the logical statements being imposed;
2. their grounding in the SwinUNETR outputs;
3. the Focal-logLTN quantifier and its negative-log loss;
4. the implementation-specific reductions over voxels, sides, and patients;
5. the external Dice objective with which the constraint is trained.

With one shared focal exponent, the most accurate name for the implemented
auxiliary objective is:

> **a guarded, side-balanced, patient-balanced Focal-logLTN boundary
> regularizer.**

It is a direct instance of the mean Focal-logLTN universal aggregation for two
guarded unary formulas. The complete training objective is hybrid rather than
purely logical because it adds this auxiliary negative log-satisfaction to the
ordinary multiclass Dice loss.

The implementation default is `focal_gamma=0`. That setting is the standard
mean logLTN/BCE boundary loss. An experimental run should be called
**Focal-logLTN** only when its recorded effective shared exponent satisfies
$\gamma>0$; the gamma value is bound into the calibration and run provenance.

The published reference for the focal quantifier is:

> Luca Piano, Francesco Manigrasso, Alessandro Russo, and Lia Morra,
> “Enhancing Neuro-Symbolic Integration with Focal Loss: A Study on Logic
> Tensor Networks,” NeSy 2024, LNCS 14980.
> [DOI: 10.1007/978-3-031-71170-1_2](https://doi.org/10.1007/978-3-031-71170-1_2).

## LTN primer: symbols, predicates, and grounding

An LTN combines a symbolic first-order language with differentiable tensor
operations. The symbolic layer states *what should be true*; the grounding
layer specifies *how each symbol is represented and evaluated numerically*.
Training changes the parameters of the grounded predicates so that the
grounded formulas become more satisfied.

### Classical logic versus LTN semantics

In classical first-order logic, a predicate such as $H(v)$ is either true or
false:

$$
H(v)\in\{0,1\}.
$$

In an LTN, the predicate is fuzzy and differentiable:

$$
\mathcal G_\theta(H(v))\in[0,1].
$$

A value near one means that the current model strongly satisfies the atom; a
value near zero means that it strongly violates it. Intermediate values encode
graded truth. In this experiment, the graded truth is taken from the model's
foreground probability.

The main correspondence is:

| First-order logic concept | LTN realization | This experiment |
|---|---|---|
| Individual | Tensor-represented object | A voxel together with its patient/image context |
| Variable | Symbol ranging over individuals | $v$ ranging over voxels in one guard domain |
| Constant | Symbol denoting one fixed individual | Not needed explicitly in the implemented formulas |
| Function symbol | Map from individuals to individuals/tensors | SwinUNETR can be viewed as a tensor function producing voxel logits |
| Predicate symbol | Relation returning truth | $H$: “is hippocampus foreground” |
| Atomic formula | Predicate applied to terms | $H(v)$ or $\neg H(v)$ |
| Connective | Fuzzy truth operation | $\neg t=1-t$ |
| Universal quantifier | Aggregation over grounded atoms | Mean Focal-logLTN aggregation over a band |
| Knowledge base | Collection of formulas | The inner and outer boundary formulas |
| Learning | Increase formula satisfaction | Minimize negative log-satisfaction together with Dice |

### What is a predicate?

A predicate symbol names a property or relation. A unary predicate accepts one
individual and returns a truth value. For example,

$$
H(v):\quad\text{“voxel $v$ belongs to hippocampus foreground.”}
$$

is unary because it has one logical argument. A binary predicate would instead
describe a relation between two individuals, such as “voxel $u$ is adjacent to
voxel $v$.” No binary predicate is required by the present constraint because
adjacency is used beforehand to construct the crisp guard domains rather than
being learned as a fuzzy relation.

The symbol $H$ alone does not prescribe a neural architecture or numerical
formula. It obtains those semantics only after a grounding is chosen.

### What is the grounding map $\mathcal G$?

The calligraphic grounding map

$$
\mathcal G
$$

is the LTN interpretation of the symbolic language. It assigns a tensor,
function, or differentiable truth operation to every relevant symbol:

- a constant is grounded as a tensor representing one individual;
- a variable is grounded as a collection or batch of individuals;
- a function symbol is grounded as a tensor-valued function;
- a predicate is grounded as a function ending in $[0,1]$;
- a connective is grounded as a fuzzy truth function;
- a quantifier is grounded as an aggregation operator.

Thus $\mathcal G$ is not necessarily one neural network and should not be
confused with a single Python function. It denotes the complete numerical
interpretation. Some parts may be learned, while others may be fixed.

For this experiment, the learned part is the SwinUNETR parameterized by
$\theta$. Its outputs determine the grounding of $H$:

$$
\mathcal G_\theta(H(v))
=h_\theta(v)
=\sigma\!\left(
\operatorname{LSE}(z_{A,\theta}(v),z_{P,\theta}(v))-z_{0,\theta}(v)
\right).
$$

The remaining groundings are fixed:

$$
\mathcal G(\neg)(t)=1-t,
$$

and the universal quantifier is grounded with the mean Focal-logLTN operator
defined later in this document.

### Important notation distinction: $g_b$ is not $\mathcal G$

This document uses lowercase

$$
g_b(v)\in\{0,A,P\}
$$

for the transformed GT class label of voxel $v$ in patient $b$. This is an
ordinary label function supplied by the dataset.

Calligraphic

$$
\mathcal G
$$

is the LTN grounding map. The symbols are unrelated despite both resembling the
letter “g”:

$$
\boxed{
g_b(v)=\text{GT label},
\qquad
\mathcal G_\theta(H(v))=\text{model-grounded fuzzy truth}.
}
$$

### What is the individual represented by $v$?

Writing $H(v)$ is convenient shorthand, but the prediction at coordinate $v$
depends on the complete patient image. A formally explicit individual can be
written

$$
x_{b,v}=(X_b,v),
$$

meaning “coordinate $v$ in the context of image $X_b$.” The fully explicit
atom is then

$$
H(x_{b,v}),
$$

with grounding

$$
\mathcal G_\theta(H(x_{b,v}))=h_{b,\theta}(v).
$$

Throughout the remainder of the document, $H(v)$ keeps the patient/image
context implicit to avoid cumbersome notation.

### Variables and guarded quantification

For patient $b$, the variable $v$ is grounded by a tensorized collection of
voxel individuals. It is not quantified over an abstract infinite spatial
domain. It ranges over one finite, GT-defined set:

$$
v\in B_{\mathrm{in},b}
\quad\text{or}\quad
v\in B_{\mathrm{out},b}.
$$

The two formulas are consequently restricted quantifications:

$$
\forall_{v\in B_{\mathrm{in},b}}H(v),
\qquad
\forall_{v\in B_{\mathrm{out},b}}\neg H(v).
$$

The band masks bind the variable to the relevant individuals. They are crisp
supervision domains, not learned fuzzy predicates. One could introduce symbols
$I(v)$ and $O(v)$ and write guarded implications, but doing so would require a
choice of fuzzy implication and could change the loss. The current
implementation more faithfully corresponds to selecting the quantified
instances directly.

### From atomic truths to a formula truth

Grounding $H$ gives one truth value for every selected voxel:

$$
t_v=\mathcal G_\theta(H(v))=h_\theta(v)
$$

on the inner band, and

$$
t_v=\mathcal G_\theta(\neg H(v))=1-h_\theta(v)
$$

on the outer band.

The universal quantifier must combine these many atomic truths into one formula
satisfaction. In classical logic, a universal formula is true only if every
atom is true. A differentiable LTN replaces this discontinuous Boolean test
with a smooth aggregator. Here that aggregator is

$$
\mathcal G_\theta
\left(\forall_{v\in B}\psi(v)\right)
=\exp\left[
\frac1{|B|}\sum_{v\in B}
(1-t_v)^\gamma\log t_v
\right].
$$

The result lies in $[0,1]$ and is the Focal-logLTN satisfaction of the grounded
formula. Training minimizes its negative logarithm:

$$
-\log\mathcal G_\theta
\left(\forall_{v\in B}\psi(v)\right)
=-\frac1{|B|}\sum_{v\in B}
(1-t_v)^\gamma\log t_v.
$$

### Knowledge-base grounding in this experiment

For one patient, define

$$
\mathcal K_b=
\{\phi_{\mathrm{in},b},\phi_{\mathrm{out},b}\}.
$$

Their grounded satisfactions are

$$
S_{\mathrm{in},b}=\mathcal G_\theta(\phi_{\mathrm{in},b}),
\qquad
S_{\mathrm{out},b}=\mathcal G_\theta(\phi_{\mathrm{out},b}).
$$

The implementation assigns equal log-space importance to the two formulas:

$$
\mathcal G_\theta(\mathcal K_b)
=S_{\mathrm{case},b}
=\sqrt{S_{\mathrm{in},b}S_{\mathrm{out},b}}.
$$

Valid patient satisfactions are then combined geometrically, which is
equivalent to averaging their negative log-satisfactions. The implementation
never needs to form the potentially tiny products explicitly.

### How learning applies the logic

The symbolic formulas themselves remain fixed. What changes during training is
the parameterized grounding of $H$:

$$
\theta
\longrightarrow
z_\theta
\longrightarrow
h_\theta
\longrightarrow
\mathcal G_\theta(\phi)
\longrightarrow
L_{\mathrm{band}}.
$$

Backpropagation computes how the model parameters should change to increase the
truth of violated formulas. In the complete segmentation experiment,

$$
\theta^*
=\arg\min_\theta
\left[
L_{\mathrm{Dice}}(\theta)
+s(e)\lambda_{\mathrm{band}}L_{\mathrm{band}}(\theta)
\right].
$$

This is the central neuro-symbolic mechanism: Dice supplies conventional
supervision, while the LTN term translates explicit boundary knowledge into a
differentiable training signal. A dedicated LTN software package is not
required, because the necessary grounding and aggregation are implemented
directly with PyTorch tensor operations.

## 1. Experimental motivation

The fold-0 baseline contains 39,781 hard-label voxel errors:

- 19,512 foreground false positives;
- 16,397 foreground false negatives;
- 3,872 anterior/posterior swaps.

The boundary rule is designed for the first two groups. There are therefore

$$
19{,}512+16{,}397=35{,}909
$$

relevant foreground/background errors.

Using the exact two-step, six-connected masks employed by the loss, 35,513 of
these 35,909 errors, or 98.90%, lie on the corresponding supervised side:

- 19,176 of 19,512 false positives lie in the outer band;
- 16,337 of 16,397 false negatives lie in the inner band.

This is the direct empirical justification for the guard domains. It is more
precise than using the separate statement that 98.44% of all error types are
within Euclidean distance two of the GT outer boundary, because the latter uses
a different distance geometry and includes anterior/posterior swaps that the
foreground predicate cannot distinguish.

### 1.1 Why the canonical experiment retains Manhattan geometry

The current two iterations with a six-connected cross induce a Manhattan
radius-two neighbourhood. A matched morphological comparison on the 52 saved
fold-0 error maps gives:

| Radius-two geometry | FP/FN errors covered | FP/FN coverage | Total band voxels |
|---|---:|---:|---:|
| Manhattan | 35,513 / 35,909 | 98.90% | 300,275 |
| Euclidean | 35,591 / 35,909 | 99.11% | 315,978 |
| Chebyshev | 35,861 / 35,909 | 99.87% | 520,811 |

These are matched *morphological-radius* comparisons, not three identical
iteration procedures:

- Manhattan uses the seven-position radius-one cross twice; its effective
  radius-two structuring element contains 25 lattice offsets.
- Euclidean uses one $5^3$ radius-two ball containing the 33 offsets satisfying
  $\Delta x^2+\Delta y^2+\Delta z^2\leq 4$. Repeating a radius-one Euclidean
  lattice kernel would reproduce the Manhattan cross rather than a Euclidean
  radius-two ball.
- Chebyshev uses the full $3^3$ cube twice; its effective radius-two
  structuring element is the complete 125-position $5^3$ cube.

Manhattan therefore retains a conservative and clearly defined supervision
region. Relative to it, Euclidean geometry adds 15,703 band voxels to cover 78
additional FP/FN errors, while Chebyshev adds 220,536 band voxels, a 73.4%
increase, to cover 348 additional FP/FN errors. This is a geometric and
empirical design choice, not a claim that Manhattan distance has inherently
more stable gradients.

## 2. Domain and preprocessing

For the canonical experiment, each image and its GT label are jointly
transformed to the final model grid

$$
\Omega=\{1,\ldots,64\}^3.
$$

The default non-resize data path pads dimensions smaller than 64 and
centre-crops dimensions larger than 64. Image and label undergo the same
spatial transformation, so their coordinates remain aligned. A subsequent
divisibility pad ensures compatibility with the SwinUNETR hierarchy; it is a
no-op when the dimensions are already $64^3$.

The bands are constructed **after** these transformations, from the label that
is actually compared with the final model logits. Consequently, “two voxels”
means two discrete steps on the transformed model grid. It does not
automatically mean two original-image voxels or two millimetres.

Let patient $b$ have transformed GT label

$$
g_b:\Omega\longrightarrow\{0,A,P\},
$$

where $0$ is background, $A$ is anterior hippocampus, and $P$ is posterior
hippocampus.

The binary GT foreground set is

$$
F_b=\{v\in\Omega:g_b(v)\in\{A,P\}\},
$$

with indicator

$$
y_{H,b}(v)=\mathbf 1[g_b(v)\in\{A,P\}].
$$

This union is intentional: the new constraint addresses foreground/background
placement and delegates the anterior/posterior distinction to the multiclass
Dice objective.

## 3. Six-connected morphological guards

For a voxel $v=(i,j,k)$, define the centre-plus-six-neighbours cross

$$
C_6(v)=\{v,(i\!\pm\!1,j,k),(i,j\!\pm\!1,k),(i,j,k\!\pm\!1)\}.
$$

Only face-sharing neighbours are used. One dilation and one erosion are

$$
D_1(F)=\{v\in\Omega:\exists u\in C_6(v),\;u\in F\},
$$

$$
E_1(F)=\{v\in\Omega:\forall u\in C_6(v),\;u\in F\}.
$$

The canonical experiment uses exactly two recursive operations:

$$
D_2(F)=D_1(D_1(F)),\qquad E_2(F)=E_1(E_1(F)).
$$

The resulting guard domains are

$$
B_{\mathrm{in},b}=F_b\setminus E_2(F_b),
$$

$$
B_{\mathrm{out},b}=D_2(F_b)\setminus F_b.
$$

They are disjoint by construction. The inner mask contains the two removed GT
foreground layers; the outer mask contains the two added GT background layers.
Deep foreground and distant background are not selected by this auxiliary
constraint.

### 3.1 Distance induced by the structuring element

Repeated use of the cross induces Manhattan distance

$$
d_1(u,v)=|u_x-v_x|+|u_y-v_y|+|u_z-v_z|.
$$

For example, offsets $(2,0,0)$ and $(1,1,0)$ are reachable in two steps, while
$(1,1,1)$ requires three. The morphology is therefore not an exact Euclidean
radius-two construction.

### 3.2 Two different kinds of padding

Dataset padding and morphological padding must not be conflated:

1. **Dataset padding** changes the image and label arrays before the model sees
   them. Added label positions have value zero and are therefore GT background.
2. **Morphological convolution padding** is a temporary one-voxel zero halo
   used on every $3^3$ convolution so that its output keeps the same shape.
   Positions outside $\Omega$ are treated as background.

If the hippocampus remains farther than two voxels from the crop edge, the
temporary halo cannot affect its bands. If foreground touches an edge, erosion
treats the missing neighbour as background and the outer band is truncated at
the grid boundary. The implementation records an `edge_touching` diagnostic for
this reason.

### 3.3 Why guard construction has no gradient

The masks are deterministic functions of the transformed GT:

$$
B_{\mathrm{in},b},B_{\mathrm{out},b}=f(g_b).
$$

They are built under `torch.no_grad` because the model must not learn or move
the guard domains. This does not stop learning from the selected voxels. For a
fixed mask $m(v)$ and differentiable voxel loss $\ell(v)$,

$$
\frac{\partial}{\partial z(v)}\bigl[m(v)\ell(v)\bigr]
=m(v)\frac{\partial\ell(v)}{\partial z(v)}.
$$

Selected voxels therefore backpropagate normally through the logits; unselected
voxels receive no gradient from this auxiliary constraint.

## 4. Grounding the hippocampus predicate

At voxel $v$, SwinUNETR produces three unrestricted logits

$$
\mathbf z_b(v)=\bigl(z_{0,b}(v),z_{A,b}(v),z_{P,b}(v)\bigr).
$$

The LTN rule needs one predicate

$$
H(v):\quad\text{“voxel $v$ belongs to hippocampus foreground.”}
$$

The categorical model probabilities are

$$
p_{c,b}(v)=
\frac{\exp z_{c,b}(v)}
{\exp z_{0,b}(v)+\exp z_{A,b}(v)+\exp z_{P,b}(v)}.
$$

Because $A$ and $P$ are disjoint outcomes of one softmax and together define
the hippocampal target, their union has model probability

$$
h_b(v)=p_{A,b}(v)+p_{P,b}(v)=1-p_{0,b}(v).
$$

This is an exact marginalization of the model's categorical distribution, not
an assumption that anterior and posterior are independent events.

### 4.1 Grouped binary logit

Define the foreground-versus-background log-odds

$$
r_{H,b}(v)=\log\frac{h_b(v)}{1-h_b(v)}.
$$

Substituting the softmax probabilities cancels their common denominator:

$$
\begin{aligned}
r_{H,b}(v)
&=\log\frac{e^{z_{A,b}(v)}+e^{z_{P,b}(v)}}{e^{z_{0,b}(v)}}\\
&=\operatorname{LSE}\bigl(z_{A,b}(v),z_{P,b}(v)\bigr)-z_{0,b}(v).
\end{aligned}
$$

The code evaluates the second line directly. It does **not** first calculate
$h/(1-h)$, so it performs no division by a potentially tiny probability.

Applying the sigmoid recovers the exact foreground probability:

$$
\sigma(r_{H,b}(v))=h_b(v).
$$

The fuzzy predicate grounding is therefore

$$
\mathcal G(H(v))=h_b(v)=\sigma(r_{H,b}(v)).
$$

With standard negation,

$$
\mathcal G(\neg H(v))=1-h_b(v)=\sigma(-r_{H,b}(v)).
$$

The grouped logit is not a second model head, introduces no parameters, and is
not itself the truth value. It is the stable scalar representation from which
the binary truth and its log-loss are evaluated.

## 5. Logical knowledge base

For each patient, the guarded knowledge base contains two unary universal
formulas:

$$
\phi_{\mathrm{in},b}
=\forall_{v\in B_{\mathrm{in},b}}H(v),
$$

$$
\phi_{\mathrm{out},b}
=\forall_{v\in B_{\mathrm{out},b}}\neg H(v).
$$

The first says that every selected voxel immediately inside the GT contour
should be foreground. The second says that every selected voxel immediately
outside it should be background.

These are **restricted quantifications** over crisp, GT-defined domains. The
implementation does not evaluate an implication such as
$B_{\mathrm{in}}(v)\rightarrow H(v)$ over every voxel. Multiplying by the mask
is the tensor implementation of selecting the quantified domain.

## 6. Standard logLTN at gamma zero

For atomic truths $t_1,\ldots,t_N\in(0,1]$, a normalized product universal
aggregator can be written

$$
S_0=\left(\prod_{i=1}^N t_i\right)^{1/N}.
$$

In log space,

$$
\log S_0=\frac1N\sum_{i=1}^N\log t_i,
$$

and its minimization loss is

$$
L_0=-\log S_0=-\frac1N\sum_{i=1}^N\log t_i.
$$

For the two boundary rules this gives

$$
L_{\mathrm{in},b}^{(0)}
=-\frac1{n_{\mathrm{in},b}}
\sum_{v\in B_{\mathrm{in},b}}\log h_b(v),
$$

$$
L_{\mathrm{out},b}^{(0)}
=-\frac1{n_{\mathrm{out},b}}
\sum_{v\in B_{\mathrm{out},b}}\log(1-h_b(v)),
$$

where $n_{\mathrm{in},b}=|B_{\mathrm{in},b}|$ and
$n_{\mathrm{out},b}=|B_{\mathrm{out},b}|$.

These are exactly binary cross-entropy with foreground target one on the inner
band and target zero on the outer band. Thus the implementation at
`focal_gamma=0` is both BCE and standard mean logLTN negative
log-satisfaction.

## 7. Published Focal-logLTN quantifier

For a universal formula with grounded atomic truths $t_i$, the mean
Focal-logLTN aggregation is

$$
\log S_\gamma
=\frac1N\sum_{i=1}^N\alpha_i(1-t_i)^\gamma\log t_i,
$$

with $\gamma\geq0$. The present implementation uses $\alpha_i=1$ for every
voxel. The optimized loss is the negative log-satisfaction

$$
L_\gamma
=-\log S_\gamma
=-\frac1N\sum_{i=1}^N(1-t_i)^\gamma\log t_i.
$$

Equivalently, in the original satisfaction domain,

$$
S_\gamma
=\prod_{i=1}^N
t_i^{(1-t_i)^\gamma/N}.
$$

At $\gamma=0$ this reduces exactly to the standard normalized product/logLTN
aggregator. For $\gamma>0$, the factor $(1-t_i)^\gamma$ suppresses atoms that
are already highly satisfied and concentrates the gradient on violated or
uncertain atoms.

## 8. Inner Focal-logLTN formula

For $v\in B_{\mathrm{in},b}$, the relevant truth is

$$
t_{\mathrm{in},b}(v)=h_b(v).
$$

The inner formula's log-satisfaction is

$$
\log S_{\mathrm{in},b}
=\frac1{n_{\mathrm{in},b}}
\sum_{v\in B_{\mathrm{in},b}}
(1-h_b(v))^\gamma\log h_b(v),
$$

and its loss is

$$
\boxed{
L_{\mathrm{in},b}
=-\frac1{n_{\mathrm{in},b}}
\sum_{v\in B_{\mathrm{in},b}}
(1-h_b(v))^\gamma\log h_b(v).
}
$$

This targets false-negative-like behaviour: GT foreground voxels whose model
truth for $H$ is low.

Using $h=\sigma(r_H)$,

$$
-\log h=-\log\sigma(r_H)=\operatorname{softplus}(-r_H).
$$

The numerically stable implemented form is therefore

$$
L_{\mathrm{in},b}
=\frac1{n_{\mathrm{in},b}}
\sum_{v\in B_{\mathrm{in},b}}
(1-\sigma(r_{H,b}(v)))^\gamma
\operatorname{softplus}(-r_{H,b}(v)).
$$

## 9. Outer Focal-logLTN formula

For $v\in B_{\mathrm{out},b}$, the formula is $\neg H(v)$ and its truth is

$$
t_{\mathrm{out},b}(v)=1-h_b(v).
$$

Its focal base is

$$
1-t_{\mathrm{out},b}(v)=h_b(v).
$$

The outer formula's log-satisfaction is

$$
\log S_{\mathrm{out},b}
=\frac1{n_{\mathrm{out},b}}
\sum_{v\in B_{\mathrm{out},b}}
h_b(v)^\gamma\log(1-h_b(v)),
$$

and its loss is

$$
\boxed{
L_{\mathrm{out},b}
=-\frac1{n_{\mathrm{out},b}}
\sum_{v\in B_{\mathrm{out},b}}
h_b(v)^\gamma\log(1-h_b(v)).
}
$$

This targets false-positive-like behaviour: GT background voxels whose model
truth for $H$ is high.

Using

$$
-\log(1-h)=-\log\sigma(-r_H)=\operatorname{softplus}(r_H),
$$

the stable form is

$$
L_{\mathrm{out},b}
=\frac1{n_{\mathrm{out},b}}
\sum_{v\in B_{\mathrm{out},b}}
\sigma(r_{H,b}(v))^\gamma
\operatorname{softplus}(r_{H,b}(v)).
$$

The relationship $\mathcal G(\neg H)=1-\mathcal G(H)$ remains exact. The focal
factor acts on the truth of the formula being aggregated; it does not redefine
negation.

## 10. Side aggregation within one patient

The two formulas are given equal importance:

$$
L_{\mathrm{case},b}
=\frac12L_{\mathrm{in},b}+\frac12L_{\mathrm{out},b}.
$$

The order of operations matters:

1. average voxel losses within the inner band;
2. average voxel losses within the outer band;
3. average the two resulting formula losses with weights $1/2$ and $1/2$.

The code does not concatenate all inner and outer voxels into one mean. If the
outer band contains more voxels, it still contributes exactly half of that
patient's auxiliary loss. Expanding the formula gives

$$
L_{\mathrm{case},b}
=\frac1{2n_{\mathrm{in},b}}
\sum_{v\in B_{\mathrm{in},b}}\ell_{\mathrm{in},b}(v)
+\frac1{2n_{\mathrm{out},b}}
\sum_{v\in B_{\mathrm{out},b}}\ell_{\mathrm{out},b}(v).
$$

Because $L=-\log S$, equal averaging in loss/log space corresponds to a
geometric aggregation of formula satisfactions:

$$
\begin{aligned}
\log S_{\mathrm{case},b}
&=\frac12\log S_{\mathrm{in},b}
+\frac12\log S_{\mathrm{out},b},\\
S_{\mathrm{case},b}
&=\sqrt{S_{\mathrm{in},b}S_{\mathrm{out},b}},\\
L_{\mathrm{case},b}
&=-\log S_{\mathrm{case},b}.
\end{aligned}
$$

The implementation optimizes `case_loss` directly. It does not multiply the
satisfactions during training.

## 11. Patient aggregation within a batch

Let $\mathcal V$ be the set of patients for which both bands are nonempty. The
batch band loss is

$$
\boxed{
L_{\mathrm{band}}
=\frac1{|\mathcal V|}
\sum_{b\in\mathcal V}L_{\mathrm{case},b}.
}
$$

Substituting the side reductions gives the complete auxiliary objective:

$$
\boxed{
L_{\mathrm{band}}
=\frac1{|\mathcal V|}
\sum_{b\in\mathcal V}
\left[
\frac1{2n_{\mathrm{in},b}}
\sum_{v\in B_{\mathrm{in},b}}\ell_{\mathrm{in},b}(v)
+\frac1{2n_{\mathrm{out},b}}
\sum_{v\in B_{\mathrm{out},b}}\ell_{\mathrm{out},b}(v)
\right].
}
$$

Every valid patient contributes equally, regardless of hippocampal or band
size. This is deliberately different from concatenating every selected voxel
from every patient into a single global mean, which would give larger patients
more influence.

In satisfaction space,

$$
\log S_{\mathrm{batch}}
=\frac1{|\mathcal V|}
\sum_{b\in\mathcal V}\log S_{\mathrm{case},b},
$$

and hence

$$
S_{\mathrm{batch}}
=\left(
\prod_{b\in\mathcal V}S_{\mathrm{case},b}
\right)^{1/|\mathcal V|}.
$$

Again, the code minimizes the equivalent mean negative log-satisfaction rather
than explicitly forming this product.

## 12. Final hybrid training objective

Let $L_{\mathrm{Dice}}$ be the ordinary three-class soft Dice loss. The
constraint module returns `bands_weight * L_band`, and the training loop applies
the epoch-dependent warm-up scale $s(e)$. The final objective is

$$
\boxed{
L_{\mathrm{total}}
=L_{\mathrm{Dice}}
+s(e)\lambda_{\mathrm{band}}L_{\mathrm{band}}.
}
$$

The responsibilities are intentionally separated:

- $L_{\mathrm{Dice}}$ supervises background, anterior, and posterior over the
  full transformed volume;
- $L_{\mathrm{band}}$ reinforces only the foreground/background transition in
  the two GT-defined guard domains;
- $L_{\mathrm{band}}$ is invariant to swapping the anterior and posterior logit
  channels, so it cannot by itself correct A/P swaps.

The auxiliary weight $\lambda_{\mathrm{band}}$ is calibrated from gradient RMS
statistics using a matching Dice-only checkpoint. A changed focal exponent,
band geometry, class grouping, or numerical policy requires a matching new
calibration rather than reuse of an old weight.

## 13. Gradient behaviour

For either side, write

$$
t=\sigma(sr_H),
$$

where $s=+1$ for the inner formula and $s=-1$ for the outer formula. The voxel
loss is

$$
\ell_\gamma(t)=-(1-t)^\gamma\log t.
$$

Its derivative with respect to the grouped logit is

$$
\boxed{
\frac{\partial\ell_\gamma}{\partial r_H}
=s(1-t)^\gamma
\left[\gamma t\log t-(1-t)\right].
}
$$

For a confidently wrong atom, $t\rightarrow0$, so

$$
\frac{\partial\ell_\gamma}{\partial r_H}\rightarrow-s.
$$

Thus a confidently wrong inner voxel receives gradient approximately $-1$ and
a confidently wrong outer voxel receives gradient approximately $+1$. The
negative log-truth therefore retains a strong corrective signal in the regime
where minimizing a raw truth error such as $1-t$ would suffer from sigmoid
saturation.

For a confidently correct atom, let $e=1-t\rightarrow0$. The gradient magnitude
is asymptotically

$$
\left|\frac{\partial\ell_\gamma}{\partial r_H}\right|
\approx(\gamma+1)e^{\gamma+1},
$$

so increasing $\gamma$ suppresses easy satisfied voxels more rapidly.

### 13.1 Gradient distribution over the three logits

Let

$$
q_A=\frac{e^{z_A}}{e^{z_A}+e^{z_P}},\qquad
q_P=\frac{e^{z_P}}{e^{z_A}+e^{z_P}}.
$$

If $g=\partial L/\partial r_H$, then

$$
\frac{\partial L}{\partial z_A}=gq_A,\qquad
\frac{\partial L}{\partial z_P}=gq_P,\qquad
\frac{\partial L}{\partial z_0}=-g.
$$

The band constraint moves probability mass between grouped foreground and
background. It does not impose an anterior-versus-posterior target; the Dice
term supplies that supervision.

## 14. Why optimization stays in log space

The implementation optimizes

$$
L_{\mathrm{case}}=-\log S_{\mathrm{case}}
$$

directly. It does not optimize $1-S_{\mathrm{case}}$. This matters because
differentiating $1-\exp(-L_{\mathrm{case}})$ would multiply the gradient by the
global satisfaction $\exp(-L_{\mathrm{case}})$, which can become tiny for a
strongly violated formula.

The reported value

$$
\texttt{truth}_b=\exp(-L_{\mathrm{case},b})
$$

is calculated after the loss for diagnostics. It is not fed back into the
optimization objective.

## 15. Numerical implementation

The implementation avoids directly evaluating unstable probability ratios or
logarithms:

1. `torch.logsumexp` computes the grouped foreground score.
2. `binary_cross_entropy_with_logits` evaluates the log-loss from $r_H$.
3. For an inner voxel,
   $$
   \operatorname{BCEWithLogits}(r_H,1)
   =-\log\sigma(r_H)=\operatorname{softplus}(-r_H).
   $$
4. For an outer voxel,
   $$
   \operatorname{BCEWithLogits}(r_H,0)
   =-\log(1-\sigma(r_H))=\operatorname{softplus}(r_H).
   $$
5. The explicit sigmoid is used for the bounded focal factor only; its output
   is not passed to `torch.log`.
6. The focal base is clamped to machine epsilon before a positive, potentially
   fractional power.
7. The entire band computation uses float32 with autocast disabled, although
   the incoming logits may have been produced by an AMP forward pass.

The code divides each side sum by $n+\epsilon$ with
$\epsilon=10^{-6}$. Since only patients with nonempty sides are optimized, this
is a negligible numerical deviation from the exact $1/n$ mean, included to
make the reduction robust.

If every patient in a batch is invalid, the implementation returns a
graph-connected float32 zero so the training loop remains differentiable and
device-consistent.

## 16. Shared versus side-specific focal exponents

For the primary, direct Focal-logLTN interpretation, use

$$
\gamma_{\mathrm{in}}=\gamma_{\mathrm{out}}=\gamma.
$$

Both formulas then use the same grounding of the published focal universal
quantifier.

The code also permits

$$
\gamma_{\mathrm{in}}\neq\gamma_{\mathrm{out}}.
$$

This does not violate
$\mathcal G(\neg H)=1-\mathcal G(H)$. It applies different focal aggregators to
the two formulas:

$$
\phi_{\mathrm{in},b}
=\forall^{F_{\gamma_{\mathrm{in}}}}_{v\in B_{\mathrm{in},b}}H(v),
$$

$$
\phi_{\mathrm{out},b}
=\forall^{F_{\gamma_{\mathrm{out}}}}_{v\in B_{\mathrm{out},b}}\neg H(v).
$$

That configuration is mathematically coherent but should be described as a
**formula-specific asymmetric Focal-logLTN extension**, not as the simplest
instance of one shared focal quantifier. Different formula weights and
different gammas also have different meanings: a weight changes a formula's
overall strength, while its gamma changes how that formula distributes gradient
between easy and hard voxels.

## 17. Interpretation of the reported satisfaction

For $\gamma=0$, the satisfaction of a formula is the normalized geometric mean
of its atomic truths.

For $\gamma>0$, Focal-logLTN gives

$$
S_\gamma
=\prod_i t_i^{(1-t_i)^\gamma/N}.
$$

This aggregator is not idempotent for intermediate truth values. If every atom
has the same truth $t\in(0,1)$, then

$$
S_\gamma=t^{(1-t)^\gamma}>t
$$

for $\gamma>0$. Consequently, `exp(-case_loss)` is a confidence-weighted focal
satisfaction diagnostic, not a calibrated probability and not directly
comparable across different gamma values. Primary comparisons across gamma
should use segmentation metrics, violation/error counts, and the explicitly
defined losses rather than interpreting this diagnostic as probability.

## 18. Exact equivalences and extensions

The following distinctions should be preserved in papers and presentations.

### Exact equivalences

- $\sigma(r_H)=p_A+p_P=1-p_0$.
- At $\gamma=0$, the voxel terms are BCE and standard mean logLTN negative
  log-satisfaction.
- With a shared $\gamma>0$ and $\alpha_i=1$, each side loss is the published
  mean Focal-logLTN universal loss.
- Equal averaging of side negative log-satisfactions corresponds to their
  weighted geometric satisfaction.
- Equal averaging of case negative log-satisfactions corresponds to a
  patient-level geometric satisfaction.

### Task-specific grounding choices

- merging anterior and posterior into one predicate;
- using two crisp GT-derived band domains;
- selecting Manhattan geometry with a six-connected cross;
- assigning equal weight to inner and outer formulas;
- assigning equal weight to valid patients;
- adding the logical regularizer to multiclass Dice;
- calibrating the external constraint weight from gradient RMS.

These choices are compatible with an LTN interpretation, but they are not
requirements of LTN logic in general.

### Explicit extension

- Separate inner and outer focal exponents are a formula-specific asymmetric
  extension of the shared-gamma formulation.

## 19. What the constraint does and does not claim

The constraint does:

- encode two explicit foreground/background boundary rules;
- focus the logLTN universal aggregation on violated atoms through focal
  modulation;
- preserve strong gradients for confidently wrong boundary predictions;
- balance the inner and outer formulas and then balance patients;
- leave the original three-class output space intact.

The constraint does not:

- learn the band locations;
- use the predicted contour to define its own supervision region;
- impose an anterior/posterior rule;
- guarantee calibrated logical truth values;
- measure physical distance in millimetres;
- require the `LTNtorch` package to instantiate the relevant tensor semantics;
- constitute a general-purpose engine for arbitrary first-order formulas.

It implements the complete grounded fragment needed for the two guarded unary
universal statements.

## 20. Code-to-mathematics map

| Mathematical object | Implementation |
|---|---|
| $F_b$ | `foreground` in `OuterBoundaryBandLoss.forward` |
| $B_{\mathrm{in}},B_{\mathrm{out}}$ | `build_boundary_bands` |
| $r_H$ | `foreground_log_odds` |
| $h=\sigma(r_H)$ | `probability` |
| $t= h$ on inner, $1-h$ on outer | `truth_probability` selected by the GT target |
| $-\log t$ | `binary_cross_entropy_with_logits(..., reduction="none")` |
| $(1-t)^\gamma$ | `focal_base.pow(gamma)` |
| $L_{\mathrm{in}},L_{\mathrm{out}}$ | masked side sums divided by side counts |
| $L_{\mathrm{case}}$ | `0.5 * (inner_loss + outer_loss)` |
| $L_{\mathrm{band}}$ | mean of `case_loss[valid]` |
| $S_{\mathrm{case}}$ diagnostic | `torch.exp(-valid_case_loss)` |
| $\lambda_{\mathrm{band}}L_{\mathrm{band}}$ | weighted result in [`objective.py`](../objective.py) |
| $L_{\mathrm{Dice}}+s(e)\lambda L_{\mathrm{band}}$ | training loop in [`train_swinunetr_constraints.py`](../train_swinunetr_constraints.py) |

The morphology and logit grouping are tested in
[`test_outer_boundary.py`](test_outer_boundary.py), and the gradient-scale
weight is produced by [`calibrate_weight.py`](calibrate_weight.py).

## 21. Compact final formulation

For every valid patient $b$:

$$
\boxed{
\begin{aligned}
B_{\mathrm{in},b}
&=F_b\setminus E_2(F_b),\\
B_{\mathrm{out},b}
&=D_2(F_b)\setminus F_b,\\[1mm]
r_{H,b}(v)
&=\operatorname{LSE}(z_{A,b}(v),z_{P,b}(v))-z_{0,b}(v),\\
h_b(v)
&=\sigma(r_{H,b}(v))=p_{A,b}(v)+p_{P,b}(v),\\[1mm]
L_{\mathrm{in},b}
&=-\frac1{n_{\mathrm{in},b}}
\sum_{v\in B_{\mathrm{in},b}}
(1-h_b(v))^\gamma\log h_b(v),\\
L_{\mathrm{out},b}
&=-\frac1{n_{\mathrm{out},b}}
\sum_{v\in B_{\mathrm{out},b}}
h_b(v)^\gamma\log(1-h_b(v)),\\
L_{\mathrm{case},b}
&=\frac12\left(L_{\mathrm{in},b}+L_{\mathrm{out},b}\right),\\
L_{\mathrm{band}}
&=\frac1{|\mathcal V|}
\sum_{b\in\mathcal V}L_{\mathrm{case},b},\\
L_{\mathrm{total}}
&=L_{\mathrm{Dice}}+s(e)\lambda_{\mathrm{band}}L_{\mathrm{band}}.
\end{aligned}
}
$$

At $\gamma=0$, $L_{\mathrm{band}}$ is a guarded mean logLTN boundary loss. At
a shared $\gamma>0$, it is the corresponding Focal-logLTN boundary loss.

## 22. Recommended thesis statement

> We ground the unary predicate $H(v)$, denoting hippocampal foreground at
> voxel $v$, with the summed anterior/posterior softmax probability
> $h(v)=p_A(v)+p_P(v)$. Two crisp, GT-derived Manhattan guard domains encode
> the formulas $\forall_{v\in B_{\mathrm{in}}}H(v)$ and
> $\forall_{v\in B_{\mathrm{out}}}\neg H(v)$. Each formula is grounded with the
> mean Focal-logLTN universal operator, its negative log-satisfaction is
> averaged within the corresponding band, the two formula losses are combined
> with equal weight for each patient, and valid patient losses are then
> averaged. The resulting side- and patient-balanced logical regularizer is
> added to the multiclass Dice loss after gradient-scale calibration and a
> linear warm-up.
