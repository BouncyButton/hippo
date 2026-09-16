# BCE-band + one-cut frozen-logit feasibility audit

Decision: **NO_GO**

The components are computationally feasible, but their matched repair does not improve on BCE bands.

## Gradient feasibility

- valid cases: 52
- median band/one-cut cosine: +0.4097
- median one-cut residual after projection on bands: 0.912
- median anchor/location cosine: +0.8262
- median location residual after projection on anchors: 0.563
- tolerance hybrid descends both components in: 100.0% of cases

## Matched counterfactual comparison versus BCE bands

### Update RMS 0.02

- union Dice: -0.000442
- 1-mm surface Dice: +0.000276
- ASSD: +0.001977 mm
- HD95: +0.000398 mm
- total errors per case: +2.981

### Update RMS 0.05

- union Dice: -0.000978
- 1-mm surface Dice: +0.000503
- ASSD: +0.004426 mm
- HD95: -0.012505 mm
- total errors per case: +6.904

### Update RMS 0.1

- union Dice: -0.001595
- 1-mm surface Dice: +0.000927
- ASSD: +0.007846 mm
- HD95: -0.006112 mm
- total errors per case: +11.442

## Gates

- PASS: `at_least_32_valid_cases`
- PASS: `simultaneous_component_descent_in_at_least_90_percent`
- PASS: `median_location_residual_at_least_20_percent`
- FAIL: `matched_counterfactual_improvement`

This is an exploratory frozen-logit diagnostic. A positive result authorizes only a matched five-epoch pilot, not a full run.
