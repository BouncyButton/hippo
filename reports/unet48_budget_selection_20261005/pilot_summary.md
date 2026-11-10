# U-Net width 48: first budget search

No qualifying reference was selected. All four complete, fresh Dice-only runs were below the final validation target of 70–75%.

| Updates | Patient-pass epochs | Training GFLOPs | Final validation macro Dice |
|---:|---:|---:|---:|
| 400 | 40.0 | 31539.940 | 22.61% |
| 800 | 80.0 | 63079.880 | 39.62% |
| 1600 | 160.0 | 126159.760 | 44.46% |
| 3200 | 320.0 | 252319.521 | 45.54% |

The selection controller stopped at its 3,200-update search bound. Training itself completed. The existing official reference remains unchanged pending investigation and a qualifying pilot.

Fold 0, seed 0; 10 training patients and 52 validation patients; 64³ inputs, batch size 1. Selection uses final macro hard Dice of anterior and posterior labels, not the best intermediate checkpoint. Search compute is separate from a future selected per-run reference.
