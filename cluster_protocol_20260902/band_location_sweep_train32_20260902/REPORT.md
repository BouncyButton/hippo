# Band-dominant cut-location residual sweep

Decision: **NO_GO**

- training cases: 32
- skipped cases: 0
- median band/location cosine: +0.4526
- median location residual after projection on bands: 0.892

## Location share 5%

Passing update magnitudes: 0/3

- RMS 0.02 (FAIL): surface Dice +0.000020, union Dice -0.000008, ASSD +0.000018 mm, errors/case +0.062
- RMS 0.05 (FAIL): surface Dice +0.000099, union Dice -0.000020, ASSD +0.000164 mm, errors/case +0.156
- RMS 0.1 (FAIL): surface Dice +0.000110, union Dice -0.000014, ASSD +0.000104 mm, errors/case +0.094

## Location share 10%

Passing update magnitudes: 0/3

- RMS 0.02 (FAIL): surface Dice +0.000042, union Dice -0.000012, ASSD +0.000031 mm, errors/case +0.062
- RMS 0.05 (FAIL): surface Dice +0.000195, union Dice -0.000041, ASSD +0.000264 mm, errors/case +0.312
- RMS 0.1 (FAIL): surface Dice +0.000147, union Dice -0.000060, ASSD +0.000273 mm, errors/case +0.438

## Location share 20%

Passing update magnitudes: 0/3

- RMS 0.02 (FAIL): surface Dice +0.000121, union Dice -0.000021, ASSD +0.000113 mm, errors/case +0.125
- RMS 0.05 (FAIL): surface Dice +0.000365, union Dice -0.000082, ASSD +0.000649 mm, errors/case +0.656
- RMS 0.1 (FAIL): surface Dice +0.000425, union Dice -0.000112, ASSD +0.000620 mm, errors/case +0.781

## Location share 30%

Passing update magnitudes: 0/3

- RMS 0.02 (FAIL): surface Dice +0.000179, union Dice -0.000031, ASSD +0.000227 mm, errors/case +0.188
- RMS 0.05 (FAIL): surface Dice +0.000466, union Dice -0.000173, ASSD +0.001036 mm, errors/case +1.312
- RMS 0.1 (FAIL): surface Dice +0.000784, union Dice -0.000256, ASSD +0.001416 mm, errors/case +1.812

No ratio passed the pre-registered robustness gate; do not train this hybrid.

This sweep uses deterministic fold-0 training cases. It does not make a held-out performance claim.
