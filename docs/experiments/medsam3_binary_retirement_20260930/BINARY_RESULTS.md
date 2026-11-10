# Retired binary MedSAM3 experiments: complete result record

Recorded before deleting binary trained adapters, predictions, probability volumes, training traces and generated plots/logs, at the user’s request on 30 September 2026. Pretrained weights, datasets, shared runtime, source and multiclass experiments are retained.

These runs merged anterior and posterior labels into a single foreground class. All Dice values here are **whole-hippocampus union Dice**, not A/P Dice. A/P swaps are undefined for these outputs.

## Study limitations

The original folds 0–2 passed individual run audits but failed the strict cross-seed calibration-consistency gate (maximum relative coefficient deviation about 0.1054%). Fold 3 reused exactly identical training-only calibration. The combined four-fold report retained the historical failure. Fold 3 was added after viewing earlier outcomes. This record retains fold 2, including its negative mean effect. Seed repetitions are not independent patients; patient identities were unavailable.

## All matched comparisons

FP/FN are mean voxel counts per validation volume; Dice is mean volume Dice.

| Run | Volumes | Baseline Dice (%) | Bands Dice (%) | Bands+edge Dice (%) | Combined gain (pp) |
|---|---:|---:|---:|---:|---:|
| pilot | 5 | 86.507952 | 87.050153 | 87.759908 | +1.251956 |
| fold0_seed17 | 46 | 83.545252 | 84.910653 | 86.545111 | +2.999859 |
| fold0_seed83 | 46 | 82.350601 | 83.495675 | 85.118831 | +2.768230 |
| fold0_seed191 | 46 | 82.158629 | 82.014899 | 84.566863 | +2.408234 |
| fold1_seed17 | 52 | 83.561916 | 83.868101 | 84.430334 | +0.868418 |
| fold1_seed83 | 52 | 82.132798 | 83.259569 | 85.006569 | +2.873771 |
| fold1_seed191 | 52 | 83.470415 | 83.670644 | 82.584676 | -0.885739 |
| fold2_seed17 | 47 | 81.224177 | 80.337232 | 80.670684 | -0.553493 |
| fold2_seed83 | 47 | 83.945725 | 83.362557 | 84.039921 | +0.094196 |
| fold2_seed191 | 47 | 83.779111 | 80.882886 | 82.948321 | -0.830791 |
| fold3_seed17 | 44 | 85.241004 | 85.579544 | 85.635468 | +0.394465 |
| fold3_seed83 | 44 | 84.429134 | 84.965963 | 85.035444 | +0.606310 |
| fold3_seed191 | 44 | 84.113819 | 83.530975 | 85.279590 | +1.165771 |

## Fold means across seeds 17, 83, 191

| Fold | Arm | Dice (%) | FP/volume | FN/volume |
|---|---|---:|---:|---:|
| 0 | baseline | 82.684827 | 487.434783 | 649.905797 |
| 0 | bands | 83.473742 | 542.920290 | 566.108696 |
| 0 | bands_edge | 85.410268 | 517.550725 | 468.173913 |
| 1 | baseline | 83.055043 | 408.801282 | 677.314103 |
| 1 | bands | 83.599438 | 507.948718 | 575.467949 |
| 1 | bands_edge | 84.007193 | 552.884615 | 521.673077 |
| 2 | baseline | 82.983004 | 589.666667 | 510.184397 |
| 2 | bands | 81.527558 | 802.609929 | 450.106383 |
| 2 | bands_edge | 82.552975 | 789.709220 | 398.730496 |
| 3 | baseline | 84.594652 | 440.189394 | 553.015152 |
| 3 | bands | 84.692161 | 496.583333 | 506.371212 |
| 3 | bands_edge | 85.316834 | 551.340909 | 429.060606 |

## Per-volume results and provenance

### pilot

Source: `/mnt/beegfsstudents/home/3160552/medsam3_supervised_5shot_20260928/results_673386`

Original results SHA256: `d556b6ad90e82a613a34f7c205e551d0ab2bcf0fb4d2d66f568cc3da7a287172`

Independent run audit: **passed**. Training seed: 42; updates per arm: 200.

Support volumes: hippocampus_011, hippocampus_056, hippocampus_146, hippocampus_319, hippocampus_363.

| Case | Arm | Dice (%) | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| hippocampus_017 | baseline | 88.26731176 | 2919 | 217 | 559 |
| hippocampus_017 | bands | 89.20300752 | 2966 | 206 | 512 |
| hippocampus_017 | bands_edge | 89.49301984 | 3045 | 282 | 433 |
| hippocampus_019 | baseline | 84.41950798 | 2934 | 661 | 422 |
| hippocampus_019 | bands | 85.70595099 | 2938 | 562 | 418 |
| hippocampus_019 | bands_edge | 86.67239897 | 3024 | 598 | 332 |
| hippocampus_033 | baseline | 88.62638195 | 2926 | 254 | 497 |
| hippocampus_033 | bands | 88.98266767 | 2952 | 260 | 471 |
| hippocampus_033 | bands_edge | 89.40416789 | 3046 | 345 | 377 |
| hippocampus_035 | baseline | 87.95603725 | 2881 | 220 | 569 |
| hippocampus_035 | bands | 87.64350453 | 2901 | 269 | 549 |
| hippocampus_035 | bands_edge | 89.00615655 | 3036 | 336 | 414 |
| hippocampus_037 | baseline | 83.27052269 | 2541 | 367 | 654 |
| hippocampus_037 | bands | 83.71563364 | 2573 | 379 | 622 |
| hippocampus_037 | bands_edge | 84.22379827 | 2672 | 478 | 523 |

### fold0_seed17

Source: `/mnt/beegfsstudents/home/3160552/medsam3_replication_20260928/results/fold0_seed17`

Original results SHA256: `23d113051110209b70e8d93e2fe0e834babd64bc0d48db6ef57f67ae37a2582e`

Independent run audit: **passed**. Training seed: 17; updates per arm: 200.

Support volumes: hippocampus_156, hippocampus_197, hippocampus_260, hippocampus_335, hippocampus_370.

| Case | Arm | Dice (%) | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| hippocampus_049 | baseline | 83.95098184 | 2843 | 202 | 885 |
| hippocampus_049 | bands | 84.91197439 | 2918 | 227 | 810 |
| hippocampus_049 | bands_edge | 87.64424430 | 3114 | 264 | 614 |
| hippocampus_052 | baseline | 81.67002949 | 2631 | 451 | 730 |
| hippocampus_052 | bands | 82.92906178 | 2718 | 476 | 643 |
| hippocampus_052 | bands_edge | 88.51800340 | 2864 | 246 | 497 |
| hippocampus_065 | baseline | 85.07997794 | 3085 | 517 | 565 |
| hippocampus_065 | bands | 88.44543678 | 3169 | 347 | 481 |
| hippocampus_065 | bands_edge | 90.55053410 | 3306 | 346 | 344 |
| hippocampus_083 | baseline | 83.20337882 | 2561 | 224 | 810 |
| hippocampus_083 | bands | 84.70738319 | 2656 | 244 | 715 |
| hippocampus_083 | bands_edge | 88.47500796 | 2779 | 132 | 592 |
| hippocampus_088 | baseline | 84.14481898 | 3033 | 298 | 845 |
| hippocampus_088 | bands | 85.59413999 | 3155 | 339 | 723 |
| hippocampus_088 | bands_edge | 86.88023395 | 3268 | 377 | 610 |
| hippocampus_090 | baseline | 85.31526549 | 3085 | 146 | 916 |
| hippocampus_090 | bands | 87.04255896 | 3211 | 166 | 790 |
| hippocampus_090 | bands_edge | 88.63004633 | 3348 | 206 | 653 |
| hippocampus_092 | baseline | 89.23076923 | 2697 | 206 | 445 |
| hippocampus_092 | bands | 89.63275918 | 2758 | 254 | 384 |
| hippocampus_092 | bands_edge | 90.44001266 | 2857 | 319 | 285 |
| hippocampus_107 | baseline | 85.09394572 | 3057 | 182 | 889 |
| hippocampus_107 | bands | 87.76394153 | 3242 | 200 | 704 |
| hippocampus_107 | bands_edge | 88.39545334 | 3344 | 276 | 602 |
| hippocampus_108 | baseline | 87.45039004 | 3195 | 172 | 745 |
| hippocampus_108 | bands | 89.23606514 | 3370 | 243 | 570 |
| hippocampus_108 | bands_edge | 89.99741535 | 3482 | 316 | 458 |
| hippocampus_123 | baseline | 83.13671062 | 2396 | 139 | 833 |
| hippocampus_123 | bands | 83.11332076 | 2424 | 180 | 805 |
| hippocampus_123 | bands_edge | 86.80476268 | 2661 | 241 | 568 |
| hippocampus_125 | baseline | 85.20599251 | 2275 | 339 | 451 |
| hippocampus_125 | bands | 84.38709677 | 2289 | 410 | 437 |
| hippocampus_125 | bands_edge | 82.81250000 | 2279 | 499 | 447 |
| hippocampus_157 | baseline | 77.12045617 | 2705 | 958 | 647 |
| hippocampus_157 | bands | 79.15643507 | 2740 | 831 | 612 |
| hippocampus_157 | bands_edge | 80.16759777 | 2870 | 938 | 482 |
| hippocampus_164 | baseline | 70.44552272 | 3194 | 2005 | 675 |
| hippocampus_164 | bands | 72.15064420 | 3276 | 1936 | 593 |
| hippocampus_164 | bands_edge | 76.06997469 | 3457 | 1763 | 412 |
| hippocampus_169 | baseline | 77.82371424 | 2353 | 835 | 506 |
| hippocampus_169 | bands | 81.21963478 | 2424 | 686 | 435 |
| hippocampus_169 | bands_edge | 84.49483161 | 2534 | 605 | 325 |
| hippocampus_175 | baseline | 84.99222395 | 2186 | 219 | 553 |
| hippocampus_175 | bands | 86.96494286 | 2245 | 179 | 494 |
| hippocampus_175 | bands_edge | 88.74196511 | 2416 | 290 | 323 |
| hippocampus_185 | baseline | 86.45666489 | 2429 | 77 | 684 |
| hippocampus_185 | bands | 87.88939447 | 2511 | 90 | 602 |
| hippocampus_185 | bands_edge | 87.68768769 | 2628 | 253 | 485 |
| hippocampus_190 | baseline | 88.30731558 | 2885 | 340 | 424 |
| hippocampus_190 | bands | 88.08179460 | 2886 | 358 | 423 |
| hippocampus_190 | bands_edge | 88.19506393 | 2966 | 451 | 343 |
| hippocampus_194 | baseline | 85.51650692 | 2409 | 331 | 485 |
| hippocampus_194 | bands | 88.87685085 | 2461 | 183 | 433 |
| hippocampus_194 | bands_edge | 90.17361111 | 2597 | 269 | 297 |
| hippocampus_204 | baseline | 86.77519663 | 2372 | 222 | 501 |
| hippocampus_204 | bands | 88.46975089 | 2486 | 261 | 387 |
| hippocampus_204 | bands_edge | 90.80682018 | 2583 | 233 | 290 |
| hippocampus_205 | baseline | 83.37055838 | 2053 | 168 | 651 |
| hippocampus_205 | bands | 83.19493871 | 2104 | 250 | 600 |
| hippocampus_205 | bands_edge | 84.47914683 | 2218 | 329 | 486 |
| hippocampus_210 | baseline | 79.70588235 | 2439 | 686 | 556 |
| hippocampus_210 | bands | 81.00032478 | 2494 | 669 | 501 |
| hippocampus_210 | bands_edge | 86.98847503 | 2604 | 388 | 391 |
| hippocampus_217 | baseline | 86.22484386 | 2485 | 177 | 617 |
| hippocampus_217 | bands | 86.18132794 | 2557 | 275 | 545 |
| hippocampus_217 | bands_edge | 85.50174106 | 2701 | 515 | 401 |
| hippocampus_219 | baseline | 80.47779022 | 2156 | 372 | 674 |
| hippocampus_219 | bands | 84.09090909 | 2220 | 230 | 610 |
| hippocampus_219 | bands_edge | 87.38673432 | 2411 | 277 | 419 |
| hippocampus_229 | baseline | 85.21962167 | 2658 | 403 | 519 |
| hippocampus_229 | bands | 87.90503689 | 2740 | 317 | 437 |
| hippocampus_229 | bands_edge | 89.38340287 | 2892 | 402 | 285 |
| hippocampus_244 | baseline | 82.94825903 | 2549 | 389 | 659 |
| hippocampus_244 | bands | 83.49079263 | 2607 | 430 | 601 |
| hippocampus_244 | bands_edge | 85.57602248 | 2741 | 457 | 467 |
| hippocampus_261 | baseline | 85.46244251 | 3345 | 412 | 726 |
| hippocampus_261 | bands | 85.53334161 | 3444 | 538 | 627 |
| hippocampus_261 | bands_edge | 86.88422881 | 3501 | 487 | 570 |
| hippocampus_264 | baseline | 83.82000282 | 2971 | 276 | 871 |
| hippocampus_264 | bands | 85.23862375 | 3072 | 294 | 770 |
| hippocampus_264 | bands_edge | 85.98309245 | 3153 | 339 | 689 |
| hippocampus_277 | baseline | 89.50451164 | 2827 | 186 | 477 |
| hippocampus_277 | bands | 90.28912634 | 2873 | 187 | 431 |
| hippocampus_277 | bands_edge | 91.94094125 | 2989 | 209 | 315 |
| hippocampus_280 | baseline | 72.87173666 | 1926 | 747 | 687 |
| hippocampus_280 | bands | 79.44609298 | 2008 | 434 | 605 |
| hippocampus_280 | bands_edge | 80.19940179 | 2011 | 391 | 602 |
| hippocampus_286 | baseline | 84.46045522 | 2579 | 356 | 593 |
| hippocampus_286 | bands | 85.65015232 | 2671 | 394 | 501 |
| hippocampus_286 | bands_edge | 87.54809912 | 2844 | 481 | 328 |
| hippocampus_288 | baseline | 79.83739837 | 2946 | 589 | 899 |
| hippocampus_288 | bands | 84.76692564 | 3055 | 308 | 790 |
| hippocampus_288 | bands_edge | 86.06568833 | 3079 | 231 | 766 |
| hippocampus_289 | baseline | 86.12873980 | 2375 | 402 | 363 |
| hippocampus_289 | bands | 85.76474744 | 2428 | 496 | 310 |
| hippocampus_289 | bands_edge | 86.85495119 | 2491 | 507 | 247 |
| hippocampus_296 | baseline | 84.14684449 | 3060 | 295 | 858 |
| hippocampus_296 | bands | 84.84601818 | 3127 | 326 | 791 |
| hippocampus_296 | bands_edge | 86.03425560 | 3265 | 407 | 653 |
| hippocampus_305 | baseline | 81.31438721 | 2289 | 312 | 740 |
| hippocampus_305 | bands | 79.94477045 | 2316 | 449 | 713 |
| hippocampus_305 | bands_edge | 81.58382477 | 2421 | 485 | 608 |
| hippocampus_308 | baseline | 83.90990726 | 2850 | 289 | 804 |
| hippocampus_308 | bands | 85.98588506 | 2985 | 304 | 669 |
| hippocampus_308 | bands_edge | 86.96622942 | 3116 | 396 | 538 |
| hippocampus_317 | baseline | 85.53064744 | 2728 | 366 | 557 |
| hippocampus_317 | bands | 85.81571230 | 2780 | 414 | 505 |
| hippocampus_317 | bands_edge | 87.23855714 | 2878 | 435 | 407 |
| hippocampus_327 | baseline | 89.24792679 | 3121 | 230 | 522 |
| hippocampus_327 | bands | 89.42470389 | 3171 | 278 | 472 |
| hippocampus_327 | bands_edge | 90.09995892 | 3290 | 370 | 353 |
| hippocampus_330 | baseline | 75.01046463 | 2688 | 482 | 1309 |
| hippocampus_330 | bands | 75.74468085 | 2759 | 529 | 1238 |
| hippocampus_330 | bands_edge | 76.53225806 | 2847 | 596 | 1150 |
| hippocampus_332 | baseline | 82.84291535 | 2745 | 555 | 582 |
| hippocampus_332 | bands | 87.88546256 | 2793 | 236 | 534 |
| hippocampus_332 | bands_edge | 88.10397554 | 2881 | 332 | 446 |
| hippocampus_338 | baseline | 84.57126470 | 2949 | 402 | 674 |
| hippocampus_338 | bands | 85.12676056 | 3022 | 455 | 601 |
| hippocampus_338 | bands_edge | 85.88923663 | 3156 | 570 | 467 |
| hippocampus_349 | baseline | 84.93698740 | 2123 | 286 | 467 |
| hippocampus_349 | bands | 83.93099438 | 2165 | 404 | 425 |
| hippocampus_349 | bands_edge | 85.31362517 | 2251 | 436 | 339 |
| hippocampus_350 | baseline | 82.81084889 | 2351 | 385 | 591 |
| hippocampus_350 | bands | 83.42997716 | 2374 | 375 | 568 |
| hippocampus_350 | bands_edge | 84.85370052 | 2465 | 403 | 477 |
| hippocampus_356 | baseline | 84.81291269 | 2890 | 429 | 606 |
| hippocampus_356 | bands | 85.43632932 | 2942 | 449 | 554 |
| hippocampus_356 | bands_edge | 87.39061828 | 3046 | 429 | 450 |
| hippocampus_358 | baseline | 79.77781635 | 2298 | 576 | 589 |
| hippocampus_358 | bands | 79.57568614 | 2363 | 689 | 524 |
| hippocampus_358 | bands_edge | 83.01886792 | 2464 | 585 | 423 |
| hippocampus_374 | baseline | 87.42399116 | 3163 | 196 | 714 |
| hippocampus_374 | bands | 88.55003404 | 3252 | 216 | 625 |
| hippocampus_374 | bands_edge | 89.47854785 | 3389 | 309 | 488 |
| hippocampus_394 | baseline | 85.79258618 | 2974 | 145 | 840 |
| hippocampus_394 | bands | 87.06749858 | 3070 | 168 | 744 |
| hippocampus_394 | bands_edge | 88.29376035 | 3198 | 232 | 616 |

### fold0_seed83

Source: `/mnt/beegfsstudents/home/3160552/medsam3_replication_20260928/results/fold0_seed83`

Original results SHA256: `b3065a4662084c99ebd72915c71e3053b8d28cc1ca33206db8f0677c307aa9a1`

Independent run audit: **passed**. Training seed: 83; updates per arm: 200.

Support volumes: hippocampus_156, hippocampus_197, hippocampus_260, hippocampus_335, hippocampus_370.

| Case | Arm | Dice (%) | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| hippocampus_049 | baseline | 86.01719198 | 3002 | 250 | 726 |
| hippocampus_049 | bands | 86.77697638 | 3068 | 275 | 660 |
| hippocampus_049 | bands_edge | 86.58785161 | 3186 | 445 | 542 |
| hippocampus_052 | baseline | 79.74201114 | 2720 | 741 | 641 |
| hippocampus_052 | bands | 78.35080759 | 2765 | 932 | 596 |
| hippocampus_052 | bands_edge | 80.54273325 | 2879 | 909 | 482 |
| hippocampus_065 | baseline | 80.60078253 | 3193 | 1080 | 457 |
| hippocampus_065 | bands | 85.92162554 | 3256 | 673 | 394 |
| hippocampus_065 | bands_edge | 87.28127448 | 3342 | 666 | 308 |
| hippocampus_083 | baseline | 81.74352495 | 2588 | 373 | 783 |
| hippocampus_083 | bands | 84.47481328 | 2658 | 264 | 713 |
| hippocampus_083 | bands_edge | 86.06079214 | 2803 | 340 | 568 |
| hippocampus_088 | baseline | 86.53158522 | 3267 | 406 | 611 |
| hippocampus_088 | bands | 87.14341540 | 3345 | 454 | 533 |
| hippocampus_088 | bands_edge | 87.36895935 | 3417 | 527 | 461 |
| hippocampus_090 | baseline | 82.83393261 | 3110 | 398 | 891 |
| hippocampus_090 | bands | 85.39657020 | 3187 | 276 | 814 |
| hippocampus_090 | bands_edge | 88.09272919 | 3344 | 247 | 657 |
| hippocampus_092 | baseline | 89.44325128 | 2707 | 204 | 435 |
| hippocampus_092 | bands | 89.96246124 | 2756 | 229 | 386 |
| hippocampus_092 | bands_edge | 90.47923323 | 2832 | 286 | 310 |
| hippocampus_107 | baseline | 82.34205657 | 3115 | 505 | 831 |
| hippocampus_107 | bands | 84.63042628 | 3246 | 479 | 700 |
| hippocampus_107 | bands_edge | 88.88888889 | 3348 | 239 | 598 |
| hippocampus_108 | baseline | 88.39572193 | 3306 | 234 | 634 |
| hippocampus_108 | bands | 89.12670102 | 3373 | 256 | 567 |
| hippocampus_108 | bands_edge | 89.16274933 | 3509 | 422 | 431 |
| hippocampus_123 | baseline | 85.21446594 | 2533 | 183 | 696 |
| hippocampus_123 | bands | 84.38008461 | 2593 | 324 | 636 |
| hippocampus_123 | bands_edge | 86.56088083 | 2673 | 274 | 556 |
| hippocampus_125 | baseline | 84.02689442 | 2312 | 465 | 414 |
| hippocampus_125 | bands | 85.38529672 | 2410 | 509 | 316 |
| hippocampus_125 | bands_edge | 85.69933729 | 2457 | 551 | 269 |
| hippocampus_157 | baseline | 79.25776755 | 2755 | 845 | 597 |
| hippocampus_157 | bands | 81.14219452 | 2799 | 748 | 553 |
| hippocampus_157 | bands_edge | 83.13579894 | 2911 | 740 | 441 |
| hippocampus_164 | baseline | 71.13847483 | 3293 | 2096 | 576 |
| hippocampus_164 | bands | 71.94321970 | 3345 | 2085 | 524 |
| hippocampus_164 | bands_edge | 77.39549479 | 3453 | 1601 | 416 |
| hippocampus_169 | baseline | 76.52173913 | 2376 | 975 | 483 |
| hippocampus_169 | bands | 73.98005440 | 2448 | 1311 | 411 |
| hippocampus_169 | bands_edge | 77.30024213 | 2554 | 1195 | 305 |
| hippocampus_175 | baseline | 85.06543495 | 2210 | 247 | 529 |
| hippocampus_175 | bands | 85.41432321 | 2278 | 317 | 461 |
| hippocampus_175 | bands_edge | 87.72455090 | 2344 | 261 | 395 |
| hippocampus_185 | baseline | 85.24927189 | 2488 | 236 | 625 |
| hippocampus_185 | bands | 88.66264021 | 2569 | 113 | 544 |
| hippocampus_185 | bands_edge | 87.56493506 | 2697 | 350 | 416 |
| hippocampus_190 | baseline | 88.92274874 | 2918 | 336 | 391 |
| hippocampus_190 | bands | 89.06250000 | 2964 | 383 | 345 |
| hippocampus_190 | bands_edge | 87.56132756 | 3034 | 587 | 275 |
| hippocampus_194 | baseline | 82.74563506 | 2417 | 531 | 477 |
| hippocampus_194 | bands | 83.79287155 | 2492 | 562 | 402 |
| hippocampus_194 | bands_edge | 84.59124690 | 2561 | 600 | 333 |
| hippocampus_204 | baseline | 87.71678884 | 2453 | 267 | 420 |
| hippocampus_204 | bands | 88.39584434 | 2510 | 296 | 363 |
| hippocampus_204 | bands_edge | 89.02606310 | 2596 | 363 | 277 |
| hippocampus_205 | baseline | 84.10958904 | 2149 | 257 | 555 |
| hippocampus_205 | bands | 83.92484342 | 2211 | 354 | 493 |
| hippocampus_205 | bands_edge | 84.86426181 | 2282 | 392 | 422 |
| hippocampus_210 | baseline | 78.36349331 | 2490 | 870 | 505 |
| hippocampus_210 | bands | 78.89736801 | 2533 | 893 | 462 |
| hippocampus_210 | bands_edge | 81.63970473 | 2599 | 773 | 396 |
| hippocampus_217 | baseline | 82.42016807 | 2452 | 396 | 650 |
| hippocampus_217 | bands | 85.41078838 | 2573 | 350 | 529 |
| hippocampus_217 | bands_edge | 86.50793651 | 2616 | 330 | 486 |
| hippocampus_219 | baseline | 81.50096440 | 2324 | 549 | 506 |
| hippocampus_219 | bands | 83.45529181 | 2338 | 435 | 492 |
| hippocampus_219 | bands_edge | 87.41746148 | 2383 | 239 | 447 |
| hippocampus_229 | baseline | 85.91304348 | 2717 | 431 | 460 |
| hippocampus_229 | bands | 88.47690587 | 2768 | 312 | 409 |
| hippocampus_229 | bands_edge | 89.47531822 | 2882 | 383 | 295 |
| hippocampus_244 | baseline | 82.68858801 | 2565 | 431 | 643 |
| hippocampus_244 | bands | 82.66160898 | 2615 | 504 | 593 |
| hippocampus_244 | bands_edge | 84.90326673 | 2677 | 421 | 531 |
| hippocampus_261 | baseline | 81.18691475 | 3338 | 814 | 733 |
| hippocampus_261 | bands | 85.41309144 | 3484 | 603 | 587 |
| hippocampus_261 | bands_edge | 87.36983952 | 3566 | 526 | 505 |
| hippocampus_264 | baseline | 83.31296673 | 3068 | 455 | 774 |
| hippocampus_264 | bands | 83.10160428 | 3108 | 530 | 734 |
| hippocampus_264 | bands_edge | 84.27987877 | 3198 | 549 | 644 |
| hippocampus_277 | baseline | 90.00807103 | 2788 | 103 | 516 |
| hippocampus_277 | bands | 90.94943731 | 2869 | 136 | 435 |
| hippocampus_277 | bands_edge | 91.32405815 | 2921 | 172 | 383 |
| hippocampus_280 | baseline | 74.31175934 | 2038 | 834 | 575 |
| hippocampus_280 | bands | 74.18397626 | 2000 | 779 | 613 |
| hippocampus_280 | bands_edge | 78.39122486 | 2144 | 713 | 469 |
| hippocampus_286 | baseline | 82.14670152 | 2671 | 660 | 501 |
| hippocampus_286 | bands | 83.09685379 | 2694 | 618 | 478 |
| hippocampus_286 | bands_edge | 85.04055272 | 2831 | 655 | 341 |
| hippocampus_288 | baseline | 78.23880218 | 3083 | 953 | 762 |
| hippocampus_288 | bands | 78.35077230 | 3145 | 1038 | 700 |
| hippocampus_288 | bands_edge | 83.00983111 | 3293 | 796 | 552 |
| hippocampus_289 | baseline | 85.41294244 | 2389 | 467 | 349 |
| hippocampus_289 | bands | 86.27172756 | 2432 | 468 | 306 |
| hippocampus_289 | bands_edge | 87.62071993 | 2495 | 462 | 243 |
| hippocampus_296 | baseline | 81.92225772 | 3077 | 517 | 841 |
| hippocampus_296 | bands | 81.99554799 | 3131 | 588 | 787 |
| hippocampus_296 | bands_edge | 81.87936508 | 3224 | 733 | 694 |
| hippocampus_305 | baseline | 77.10144928 | 2261 | 575 | 768 |
| hippocampus_305 | bands | 81.02938619 | 2330 | 392 | 699 |
| hippocampus_305 | bands_edge | 81.21274059 | 2384 | 458 | 645 |
| hippocampus_308 | baseline | 74.73063109 | 2913 | 1229 | 741 |
| hippocampus_308 | bands | 74.71061902 | 2969 | 1325 | 685 |
| hippocampus_308 | bands_edge | 82.00636943 | 3090 | 792 | 564 |
| hippocampus_317 | baseline | 85.32227186 | 2674 | 309 | 611 |
| hippocampus_317 | bands | 84.39727066 | 2783 | 527 | 502 |
| hippocampus_317 | bands_edge | 84.65655665 | 2847 | 594 | 438 |
| hippocampus_327 | baseline | 89.19890695 | 3101 | 209 | 542 |
| hippocampus_327 | bands | 89.78502178 | 3195 | 279 | 448 |
| hippocampus_327 | bands_edge | 90.31813800 | 3279 | 339 | 364 |
| hippocampus_330 | baseline | 74.52918392 | 2790 | 700 | 1207 |
| hippocampus_330 | bands | 76.43894567 | 2842 | 597 | 1155 |
| hippocampus_330 | bands_edge | 77.04918033 | 2914 | 653 | 1083 |
| hippocampus_332 | baseline | 85.97139643 | 2675 | 221 | 652 |
| hippocampus_332 | bands | 83.70626220 | 2787 | 545 | 540 |
| hippocampus_332 | bands_edge | 87.96324655 | 2872 | 331 | 455 |
| hippocampus_338 | baseline | 83.11253661 | 2980 | 568 | 643 |
| hippocampus_338 | bands | 84.51648966 | 3024 | 509 | 599 |
| hippocampus_338 | bands_edge | 86.64910578 | 3125 | 465 | 498 |
| hippocampus_349 | baseline | 78.19326175 | 2112 | 700 | 478 |
| hippocampus_349 | bands | 80.23730070 | 2164 | 640 | 426 |
| hippocampus_349 | bands_edge | 79.26397735 | 2240 | 822 | 350 |
| hippocampus_350 | baseline | 76.99346405 | 2356 | 822 | 586 |
| hippocampus_350 | bands | 79.55377955 | 2389 | 675 | 553 |
| hippocampus_350 | bands_edge | 81.73282820 | 2481 | 648 | 461 |
| hippocampus_356 | baseline | 78.39956745 | 2900 | 1002 | 596 |
| hippocampus_356 | bands | 81.93892921 | 2992 | 815 | 504 |
| hippocampus_356 | bands_edge | 82.08216245 | 3067 | 910 | 429 |
| hippocampus_358 | baseline | 74.34606011 | 2288 | 980 | 599 |
| hippocampus_358 | bands | 79.53196541 | 2345 | 665 | 542 |
| hippocampus_358 | bands_edge | 79.58285807 | 2442 | 808 | 445 |
| hippocampus_374 | baseline | 88.55191257 | 3241 | 202 | 636 |
| hippocampus_374 | bands | 87.32767762 | 3294 | 373 | 583 |
| hippocampus_374 | bands_edge | 89.84251969 | 3423 | 320 | 454 |
| hippocampus_394 | baseline | 86.63147382 | 3036 | 159 | 778 |
| hippocampus_394 | bands | 87.49473684 | 3117 | 194 | 697 |
| hippocampus_394 | bands_edge | 88.35804503 | 3218 | 252 | 596 |

### fold0_seed191

Source: `/mnt/beegfsstudents/home/3160552/medsam3_replication_20260928/results/fold0_seed191`

Original results SHA256: `5c79d16be8ff83d60cab74fba15df7f6e5779b06113d3508dffb012b83257530`

Independent run audit: **passed**. Training seed: 191; updates per arm: 200.

Support volumes: hippocampus_156, hippocampus_197, hippocampus_260, hippocampus_335, hippocampus_370.

| Case | Arm | Dice (%) | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| hippocampus_049 | baseline | 84.95756512 | 2903 | 203 | 825 |
| hippocampus_049 | bands | 86.56042871 | 3069 | 294 | 659 |
| hippocampus_049 | bands_edge | 87.06775373 | 3097 | 289 | 631 |
| hippocampus_052 | baseline | 80.80959520 | 2695 | 614 | 666 |
| hippocampus_052 | bands | 80.43854587 | 2788 | 783 | 573 |
| hippocampus_052 | bands_edge | 82.17737351 | 2891 | 784 | 470 |
| hippocampus_065 | baseline | 80.43933054 | 3076 | 922 | 574 |
| hippocampus_065 | bands | 75.75936651 | 3205 | 1606 | 445 |
| hippocampus_065 | bands_edge | 81.36134660 | 3311 | 1178 | 339 |
| hippocampus_083 | baseline | 84.18443233 | 2547 | 133 | 824 |
| hippocampus_083 | bands | 84.18383518 | 2656 | 283 | 715 |
| hippocampus_083 | bands_edge | 86.24396323 | 2768 | 280 | 603 |
| hippocampus_088 | baseline | 85.03893975 | 3112 | 329 | 766 |
| hippocampus_088 | bands | 87.11122826 | 3305 | 405 | 573 |
| hippocampus_088 | bands_edge | 86.76987121 | 3335 | 474 | 543 |
| hippocampus_090 | baseline | 81.92219680 | 3043 | 385 | 958 |
| hippocampus_090 | bands | 83.60827442 | 3193 | 444 | 808 |
| hippocampus_090 | bands_edge | 88.06168572 | 3312 | 209 | 689 |
| hippocampus_092 | baseline | 89.33355383 | 2701 | 204 | 441 |
| hippocampus_092 | bands | 90.02765577 | 2767 | 238 | 375 |
| hippocampus_092 | bands_edge | 89.85461441 | 2843 | 343 | 299 |
| hippocampus_107 | baseline | 81.93599577 | 3098 | 518 | 848 |
| hippocampus_107 | bands | 83.63263212 | 3260 | 590 | 686 |
| hippocampus_107 | bands_edge | 88.67008008 | 3377 | 294 | 569 |
| hippocampus_108 | baseline | 88.03315951 | 3292 | 247 | 648 |
| hippocampus_108 | bands | 89.09940306 | 3433 | 333 | 507 |
| hippocampus_108 | bands_edge | 89.74948875 | 3511 | 373 | 429 |
| hippocampus_123 | baseline | 82.64433979 | 2369 | 135 | 860 |
| hippocampus_123 | bands | 85.62343096 | 2558 | 188 | 671 |
| hippocampus_123 | bands_edge | 87.93075290 | 2692 | 202 | 537 |
| hippocampus_125 | baseline | 85.54561036 | 2246 | 279 | 480 |
| hippocampus_125 | bands | 84.54142012 | 2286 | 396 | 440 |
| hippocampus_125 | bands_edge | 84.55226543 | 2370 | 510 | 356 |
| hippocampus_157 | baseline | 78.53280725 | 2687 | 804 | 665 |
| hippocampus_157 | bands | 76.20081411 | 2808 | 1210 | 544 |
| hippocampus_157 | bands_edge | 79.34797624 | 2872 | 1015 | 480 |
| hippocampus_164 | baseline | 70.09777587 | 3262 | 2176 | 607 |
| hippocampus_164 | bands | 70.59438003 | 3379 | 2325 | 490 |
| hippocampus_164 | bands_edge | 74.18631950 | 3476 | 2026 | 393 |
| hippocampus_169 | baseline | 75.98393574 | 2365 | 1001 | 494 |
| hippocampus_169 | bands | 76.36531935 | 2475 | 1148 | 384 |
| hippocampus_169 | bands_edge | 79.03000155 | 2542 | 1032 | 317 |
| hippocampus_175 | baseline | 83.57250755 | 2213 | 344 | 526 |
| hippocampus_175 | bands | 82.12472806 | 2265 | 512 | 474 |
| hippocampus_175 | bands_edge | 88.30734967 | 2379 | 270 | 360 |
| hippocampus_185 | baseline | 85.01278772 | 2493 | 259 | 620 |
| hippocampus_185 | bands | 81.01822753 | 2578 | 673 | 535 |
| hippocampus_185 | bands_edge | 89.70612930 | 2671 | 171 | 442 |
| hippocampus_190 | baseline | 87.52126179 | 2830 | 328 | 479 |
| hippocampus_190 | bands | 88.50059737 | 2963 | 424 | 346 |
| hippocampus_190 | bands_edge | 88.81365911 | 3017 | 468 | 292 |
| hippocampus_194 | baseline | 86.18527146 | 2405 | 282 | 489 |
| hippocampus_194 | bands | 83.70531968 | 2494 | 571 | 400 |
| hippocampus_194 | bands_edge | 88.40381992 | 2592 | 378 | 302 |
| hippocampus_204 | baseline | 85.49103179 | 2407 | 351 | 466 |
| hippocampus_204 | bands | 85.33694548 | 2520 | 513 | 353 |
| hippocampus_204 | bands_edge | 88.93071001 | 2599 | 373 | 274 |
| hippocampus_205 | baseline | 85.27437821 | 2160 | 202 | 544 |
| hippocampus_205 | bands | 86.33564280 | 2243 | 249 | 461 |
| hippocampus_205 | bands_edge | 87.27754638 | 2305 | 273 | 399 |
| hippocampus_210 | baseline | 77.35257215 | 2466 | 915 | 529 |
| hippocampus_210 | bands | 77.46070502 | 2538 | 1020 | 457 |
| hippocampus_210 | bands_edge | 81.91985089 | 2637 | 806 | 358 |
| hippocampus_217 | baseline | 80.24588802 | 2415 | 502 | 687 |
| hippocampus_217 | bands | 80.61108402 | 2480 | 571 | 622 |
| hippocampus_217 | bands_edge | 82.28143993 | 2640 | 675 | 462 |
| hippocampus_219 | baseline | 79.04040404 | 2191 | 523 | 639 |
| hippocampus_219 | bands | 74.69918699 | 2297 | 1023 | 533 |
| hippocampus_219 | bands_edge | 83.72256365 | 2384 | 481 | 446 |
| hippocampus_229 | baseline | 85.14664144 | 2700 | 465 | 477 |
| hippocampus_229 | bands | 85.54842652 | 2800 | 569 | 377 |
| hippocampus_229 | bands_edge | 86.55462185 | 2884 | 603 | 293 |
| hippocampus_244 | baseline | 82.28626943 | 2541 | 427 | 667 |
| hippocampus_244 | bands | 82.19862586 | 2632 | 564 | 576 |
| hippocampus_244 | bands_edge | 82.92682927 | 2703 | 608 | 505 |
| hippocampus_261 | baseline | 84.97555956 | 3303 | 400 | 768 |
| hippocampus_261 | bands | 79.88123515 | 3363 | 986 | 708 |
| hippocampus_261 | bands_edge | 84.92396120 | 3546 | 734 | 525 |
| hippocampus_264 | baseline | 78.34854885 | 2875 | 622 | 967 |
| hippocampus_264 | bands | 82.08399787 | 3088 | 594 | 754 |
| hippocampus_264 | bands_edge | 82.82958199 | 3220 | 713 | 622 |
| hippocampus_277 | baseline | 90.03225806 | 2791 | 105 | 513 |
| hippocampus_277 | bands | 90.91770652 | 2878 | 149 | 426 |
| hippocampus_277 | bands_edge | 91.65505496 | 2960 | 195 | 344 |
| hippocampus_280 | baseline | 74.86097795 | 1952 | 650 | 661 |
| hippocampus_280 | bands | 73.31410025 | 2033 | 900 | 580 |
| hippocampus_280 | bands_edge | 73.87518142 | 2036 | 863 | 577 |
| hippocampus_286 | baseline | 84.30668842 | 2584 | 374 | 588 |
| hippocampus_286 | bands | 85.66510172 | 2737 | 481 | 435 |
| hippocampus_286 | bands_edge | 86.76696346 | 2826 | 516 | 346 |
| hippocampus_288 | baseline | 79.53094463 | 3052 | 778 | 793 |
| hippocampus_288 | bands | 79.63420662 | 3222 | 1025 | 623 |
| hippocampus_288 | bands_edge | 81.93733616 | 3282 | 884 | 563 |
| hippocampus_289 | baseline | 86.47752395 | 2347 | 343 | 391 |
| hippocampus_289 | bands | 84.41649196 | 2416 | 570 | 322 |
| hippocampus_289 | bands_edge | 85.52856654 | 2500 | 608 | 238 |
| hippocampus_296 | baseline | 79.17757988 | 3023 | 695 | 895 |
| hippocampus_296 | bands | 80.15883182 | 3129 | 760 | 789 |
| hippocampus_296 | bands_edge | 82.60703850 | 3251 | 702 | 667 |
| hippocampus_305 | baseline | 79.86600846 | 2265 | 378 | 764 |
| hippocampus_305 | bands | 74.46673034 | 2339 | 914 | 690 |
| hippocampus_305 | bands_edge | 80.14571949 | 2420 | 590 | 609 |
| hippocampus_308 | baseline | 76.84649536 | 2856 | 923 | 798 |
| hippocampus_308 | bands | 79.27974948 | 3038 | 972 | 616 |
| hippocampus_308 | bands_edge | 82.49450123 | 3188 | 887 | 466 |
| hippocampus_317 | baseline | 86.53048197 | 2711 | 270 | 574 |
| hippocampus_317 | bands | 87.09127538 | 2810 | 358 | 475 |
| hippocampus_317 | bands_edge | 87.61962631 | 2884 | 414 | 401 |
| hippocampus_327 | baseline | 89.12855910 | 3099 | 212 | 544 |
| hippocampus_327 | bands | 89.98743192 | 3222 | 296 | 421 |
| hippocampus_327 | bands_edge | 90.87680961 | 3327 | 352 | 316 |
| hippocampus_330 | baseline | 72.66464591 | 2637 | 624 | 1360 |
| hippocampus_330 | bands | 73.36097435 | 2831 | 890 | 1166 |
| hippocampus_330 | bands_edge | 74.94747899 | 2854 | 765 | 1143 |
| hippocampus_332 | baseline | 80.27598620 | 2676 | 664 | 651 |
| hippocampus_332 | bands | 80.85668182 | 2756 | 734 | 571 |
| hippocampus_332 | bands_edge | 82.70301964 | 2821 | 674 | 506 |
| hippocampus_338 | baseline | 78.71326450 | 2973 | 958 | 650 |
| hippocampus_338 | bands | 75.80405740 | 3064 | 1397 | 559 |
| hippocampus_338 | bands_edge | 83.54766008 | 3151 | 769 | 472 |
| hippocampus_349 | baseline | 77.49583256 | 2092 | 717 | 498 |
| hippocampus_349 | bands | 77.91741472 | 2170 | 810 | 420 |
| hippocampus_349 | bands_edge | 82.21083455 | 2246 | 628 | 344 |
| hippocampus_350 | baseline | 78.12865497 | 2338 | 705 | 604 |
| hippocampus_350 | bands | 78.65278230 | 2417 | 787 | 525 |
| hippocampus_350 | bands_edge | 81.17031710 | 2483 | 693 | 459 |
| hippocampus_356 | baseline | 86.75744426 | 2899 | 288 | 597 |
| hippocampus_356 | bands | 86.23063683 | 3006 | 470 | 490 |
| hippocampus_356 | bands_edge | 88.03077800 | 3089 | 433 | 407 |
| hippocampus_358 | baseline | 75.05804312 | 2263 | 880 | 624 |
| hippocampus_358 | bands | 76.80645161 | 2381 | 932 | 506 |
| hippocampus_358 | bands_edge | 77.10007880 | 2446 | 1012 | 441 |
| hippocampus_374 | baseline | 87.44160484 | 3182 | 219 | 695 |
| hippocampus_374 | bands | 87.27224716 | 3305 | 392 | 572 |
| hippocampus_374 | bands_edge | 87.77964567 | 3394 | 462 | 483 |
| hippocampus_394 | baseline | 86.09157457 | 2999 | 154 | 815 |
| hippocampus_394 | bands | 87.60100306 | 3144 | 220 | 670 |
| hippocampus_394 | bands_edge | 87.47751487 | 3161 | 252 | 653 |

### fold1_seed17

Source: `/mnt/beegfsstudents/home/3160552/medsam3_replication_20260928/results/fold1_seed17`

Original results SHA256: `515bad8cab445134e6557e696b911ac4fa467c6352ccbd09c9999b8e0b1c7dda`

Independent run audit: **passed**. Training seed: 17; updates per arm: 200.

Support volumes: hippocampus_130, hippocampus_221, hippocampus_251, hippocampus_333, hippocampus_385.

| Case | Arm | Dice (%) | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| hippocampus_003 | baseline | 83.89228482 | 2539 | 161 | 814 |
| hippocampus_003 | bands | 85.49019608 | 2725 | 297 | 628 |
| hippocampus_003 | bands_edge | 85.59458588 | 2656 | 197 | 697 |
| hippocampus_025 | baseline | 83.70286454 | 2586 | 267 | 740 |
| hippocampus_025 | bands | 84.85415692 | 2720 | 365 | 606 |
| hippocampus_025 | bands_edge | 85.65656566 | 2756 | 353 | 570 |
| hippocampus_026 | baseline | 82.32068119 | 2852 | 449 | 776 |
| hippocampus_026 | bands | 82.26415094 | 2943 | 584 | 685 |
| hippocampus_026 | bands_edge | 83.30086279 | 2993 | 565 | 635 |
| hippocampus_034 | baseline | 89.14607444 | 2850 | 169 | 525 |
| hippocampus_034 | bands | 90.86479903 | 2984 | 209 | 391 |
| hippocampus_034 | bands_edge | 89.12655971 | 3000 | 357 | 375 |
| hippocampus_036 | baseline | 86.14848821 | 2721 | 87 | 788 |
| hippocampus_036 | bands | 87.10029637 | 2792 | 110 | 717 |
| hippocampus_036 | bands_edge | 87.73279975 | 2850 | 138 | 659 |
| hippocampus_044 | baseline | 86.48559126 | 2611 | 207 | 609 |
| hippocampus_044 | bands | 86.18889416 | 2724 | 377 | 496 |
| hippocampus_044 | bands_edge | 87.58232932 | 2726 | 279 | 494 |
| hippocampus_045 | baseline | 83.29177057 | 2171 | 174 | 697 |
| hippocampus_045 | bands | 82.70462633 | 2324 | 428 | 544 |
| hippocampus_045 | bands_edge | 84.66835810 | 2336 | 314 | 532 |
| hippocampus_064 | baseline | 84.46657183 | 2969 | 401 | 691 |
| hippocampus_064 | bands | 81.78983051 | 3016 | 699 | 644 |
| hippocampus_064 | bands_edge | 80.75241754 | 3048 | 841 | 612 |
| hippocampus_087 | baseline | 85.94009983 | 3099 | 406 | 608 |
| hippocampus_087 | bands | 85.09090909 | 3159 | 559 | 548 |
| hippocampus_087 | bands_edge | 84.35710567 | 3241 | 736 | 466 |
| hippocampus_093 | baseline | 80.58097313 | 2774 | 369 | 968 |
| hippocampus_093 | bands | 80.58608059 | 2970 | 659 | 772 |
| hippocampus_093 | bands_edge | 80.95303300 | 2956 | 605 | 786 |
| hippocampus_094 | baseline | 85.56294988 | 3150 | 184 | 879 |
| hippocampus_094 | bands | 85.62572646 | 3315 | 399 | 714 |
| hippocampus_094 | bands_edge | 88.06330472 | 3283 | 144 | 746 |
| hippocampus_106 | baseline | 80.94983039 | 2267 | 242 | 825 |
| hippocampus_106 | bands | 80.68219088 | 2460 | 546 | 632 |
| hippocampus_106 | bands_edge | 82.78290025 | 2469 | 404 | 623 |
| hippocampus_114 | baseline | 84.32321455 | 2875 | 240 | 829 |
| hippocampus_114 | bands | 83.96278051 | 3068 | 536 | 636 |
| hippocampus_114 | bands_edge | 86.00328048 | 3146 | 466 | 558 |
| hippocampus_144 | baseline | 85.66516226 | 1993 | 189 | 478 |
| hippocampus_144 | bands | 86.44170540 | 2139 | 339 | 332 |
| hippocampus_144 | bands_edge | 85.96309351 | 2073 | 279 | 398 |
| hippocampus_149 | baseline | 88.90363244 | 2680 | 206 | 463 |
| hippocampus_149 | bands | 89.49607330 | 2735 | 234 | 408 |
| hippocampus_149 | bands_edge | 90.56420233 | 2793 | 232 | 350 |
| hippocampus_154 | baseline | 88.62373325 | 2711 | 150 | 546 |
| hippocampus_154 | bands | 88.66206030 | 2823 | 288 | 434 |
| hippocampus_154 | bands_edge | 88.63990040 | 2848 | 321 | 409 |
| hippocampus_155 | baseline | 82.07384132 | 2612 | 198 | 943 |
| hippocampus_155 | bands | 84.00241473 | 2783 | 288 | 772 |
| hippocampus_155 | bands_edge | 83.12126322 | 2711 | 257 | 844 |
| hippocampus_162 | baseline | 88.82265276 | 2980 | 310 | 440 |
| hippocampus_162 | bands | 87.76359202 | 3059 | 492 | 361 |
| hippocampus_162 | bands_edge | 89.22018349 | 3112 | 444 | 308 |
| hippocampus_163 | baseline | 80.62875636 | 2616 | 370 | 887 |
| hippocampus_163 | bands | 76.31615892 | 2718 | 902 | 785 |
| hippocampus_163 | bands_edge | 79.52967842 | 2807 | 749 | 696 |
| hippocampus_166 | baseline | 85.17371861 | 2476 | 311 | 551 |
| hippocampus_166 | bands | 84.91193087 | 2555 | 436 | 472 |
| hippocampus_166 | bands_edge | 84.29371090 | 2560 | 487 | 467 |
| hippocampus_171 | baseline | 84.79840433 | 2976 | 369 | 698 |
| hippocampus_171 | bands | 85.71025930 | 3041 | 381 | 633 |
| hippocampus_171 | bands_edge | 85.71428571 | 3084 | 438 | 590 |
| hippocampus_177 | baseline | 85.75112830 | 1995 | 65 | 598 |
| hippocampus_177 | bands | 87.18377587 | 2085 | 105 | 508 |
| hippocampus_177 | bands_edge | 87.73388773 | 2110 | 107 | 483 |
| hippocampus_203 | baseline | 87.33723344 | 2314 | 158 | 513 |
| hippocampus_203 | bands | 87.65652952 | 2450 | 313 | 377 |
| hippocampus_203 | bands_edge | 88.57715431 | 2431 | 231 | 396 |
| hippocampus_215 | baseline | 85.09784245 | 2544 | 102 | 789 |
| hippocampus_215 | bands | 85.80431700 | 2723 | 291 | 610 |
| hippocampus_215 | bands_edge | 84.69841270 | 2668 | 299 | 665 |
| hippocampus_223 | baseline | 84.37691835 | 2749 | 265 | 753 |
| hippocampus_223 | bands | 86.17193240 | 2932 | 371 | 570 |
| hippocampus_223 | bands_edge | 85.87520379 | 2897 | 348 | 605 |
| hippocampus_228 | baseline | 81.68389956 | 2765 | 334 | 906 |
| hippocampus_228 | bands | 81.73913043 | 2867 | 477 | 804 |
| hippocampus_228 | bands_edge | 80.37016265 | 2866 | 595 | 805 |
| hippocampus_231 | baseline | 84.25566673 | 2063 | 187 | 584 |
| hippocampus_231 | bands | 85.52552553 | 2136 | 212 | 511 |
| hippocampus_231 | bands_edge | 85.64000000 | 2141 | 212 | 506 |
| hippocampus_234 | baseline | 82.96122209 | 2471 | 270 | 745 |
| hippocampus_234 | bands | 83.12903226 | 2577 | 407 | 639 |
| hippocampus_234 | bands_edge | 84.41601050 | 2573 | 307 | 643 |
| hippocampus_235 | baseline | 80.93622795 | 2386 | 470 | 654 |
| hippocampus_235 | bands | 77.38833747 | 2495 | 913 | 545 |
| hippocampus_235 | bands_edge | 77.63699715 | 2451 | 823 | 589 |
| hippocampus_236 | baseline | 80.10600707 | 2267 | 304 | 822 |
| hippocampus_236 | bands | 83.27656676 | 2445 | 338 | 644 |
| hippocampus_236 | bands_edge | 81.92488263 | 2443 | 432 | 646 |
| hippocampus_238 | baseline | 84.53522301 | 3042 | 168 | 945 |
| hippocampus_238 | bands | 83.42663716 | 3204 | 490 | 783 |
| hippocampus_238 | bands_edge | 86.08267978 | 3238 | 298 | 749 |
| hippocampus_245 | baseline | 78.46432691 | 2381 | 363 | 944 |
| hippocampus_245 | bands | 79.39051210 | 2527 | 514 | 798 |
| hippocampus_245 | bands_edge | 79.29724596 | 2505 | 488 | 820 |
| hippocampus_250 | baseline | 84.31437027 | 2564 | 87 | 867 |
| hippocampus_250 | bands | 83.61685824 | 2728 | 366 | 703 |
| hippocampus_250 | bands_edge | 86.17153511 | 2773 | 232 | 658 |
| hippocampus_263 | baseline | 82.68411442 | 2717 | 321 | 817 |
| hippocampus_263 | bands | 82.22947100 | 2899 | 618 | 635 |
| hippocampus_263 | bands_edge | 83.09758537 | 2908 | 557 | 626 |
| hippocampus_274 | baseline | 85.21704341 | 2130 | 204 | 535 |
| hippocampus_274 | bands | 85.63162970 | 2220 | 300 | 445 |
| hippocampus_274 | bands_edge | 85.47442405 | 2189 | 268 | 476 |
| hippocampus_287 | baseline | 81.68837282 | 2603 | 70 | 1097 |
| hippocampus_287 | bands | 84.60370483 | 2786 | 100 | 914 |
| hippocampus_287 | bands_edge | 83.45390180 | 2711 | 86 | 989 |
| hippocampus_295 | baseline | 85.13755981 | 2847 | 213 | 781 |
| hippocampus_295 | bands | 82.41317282 | 2978 | 621 | 650 |
| hippocampus_295 | bands_edge | 84.77459597 | 2990 | 436 | 638 |
| hippocampus_299 | baseline | 81.09362055 | 2447 | 386 | 755 |
| hippocampus_299 | bands | 81.66507785 | 2570 | 522 | 632 |
| hippocampus_299 | bands_edge | 82.26064681 | 2569 | 475 | 633 |
| hippocampus_300 | baseline | 82.33595379 | 2566 | 263 | 838 |
| hippocampus_300 | bands | 83.51479834 | 2723 | 394 | 681 |
| hippocampus_300 | bands_edge | 84.89006705 | 2722 | 287 | 682 |
| hippocampus_309 | baseline | 81.00030497 | 2656 | 290 | 956 |
| hippocampus_309 | bands | 78.86039886 | 2768 | 640 | 844 |
| hippocampus_309 | bands_edge | 80.38818076 | 2775 | 517 | 837 |
| hippocampus_316 | baseline | 86.69514317 | 2740 | 486 | 355 |
| hippocampus_316 | bands | 86.31289188 | 2822 | 622 | 273 |
| hippocampus_316 | bands_edge | 85.95626732 | 2791 | 608 | 304 |
| hippocampus_321 | baseline | 83.36925287 | 2321 | 297 | 629 |
| hippocampus_321 | bands | 85.01898516 | 2463 | 381 | 487 |
| hippocampus_321 | bands_edge | 85.65430371 | 2448 | 318 | 502 |
| hippocampus_328 | baseline | 79.12614517 | 2807 | 331 | 1150 |
| hippocampus_328 | bands | 84.77980943 | 3292 | 517 | 665 |
| hippocampus_328 | bands_edge | 84.24764890 | 3225 | 474 | 732 |
| hippocampus_329 | baseline | 79.27927928 | 2728 | 479 | 947 |
| hippocampus_329 | bands | 79.91495393 | 2819 | 561 | 856 |
| hippocampus_329 | bands_edge | 80.21236906 | 2795 | 499 | 880 |
| hippocampus_336 | baseline | 78.22032252 | 2013 | 541 | 580 |
| hippocampus_336 | bands | 77.52995392 | 2103 | 729 | 490 |
| hippocampus_336 | bands_edge | 77.80203785 | 2138 | 765 | 455 |
| hippocampus_337 | baseline | 86.84210526 | 2607 | 144 | 646 |
| hippocampus_337 | bands | 86.49959580 | 2675 | 257 | 578 |
| hippocampus_337 | bands_edge | 86.67633317 | 2690 | 264 | 563 |
| hippocampus_345 | baseline | 83.59430605 | 2349 | 291 | 631 |
| hippocampus_345 | bands | 86.59043659 | 2499 | 293 | 481 |
| hippocampus_345 | bands_edge | 84.71720818 | 2464 | 373 | 516 |
| hippocampus_351 | baseline | 79.77806979 | 2732 | 393 | 992 |
| hippocampus_351 | bands | 80.85405254 | 2878 | 517 | 846 |
| hippocampus_351 | bands_edge | 82.98557625 | 2963 | 454 | 761 |
| hippocampus_352 | baseline | 79.75783918 | 2569 | 194 | 1110 |
| hippocampus_352 | bands | 82.15010142 | 2835 | 388 | 844 |
| hippocampus_352 | bands_edge | 83.61760069 | 2917 | 381 | 762 |
| hippocampus_354 | baseline | 79.78859122 | 2189 | 386 | 723 |
| hippocampus_354 | bands | 79.04212550 | 2261 | 548 | 651 |
| hippocampus_354 | bands_edge | 83.05932809 | 2324 | 360 | 588 |
| hippocampus_355 | baseline | 81.95395146 | 2634 | 459 | 701 |
| hippocampus_355 | bands | 82.22748815 | 2776 | 641 | 559 |
| hippocampus_355 | bands_edge | 82.48520710 | 2788 | 637 | 547 |
| hippocampus_360 | baseline | 86.33628319 | 2439 | 280 | 492 |
| hippocampus_360 | bands | 86.48460602 | 2486 | 332 | 445 |
| hippocampus_360 | bands_edge | 86.57949071 | 2516 | 365 | 415 |

### fold1_seed83

Source: `/mnt/beegfsstudents/home/3160552/medsam3_replication_20260928/results/fold1_seed83`

Original results SHA256: `11f0f410fd5f341a306923243000868b60dfd59f5790a3f6142212d468c037a7`

Independent run audit: **passed**. Training seed: 83; updates per arm: 200.

Support volumes: hippocampus_130, hippocampus_221, hippocampus_251, hippocampus_333, hippocampus_385.

| Case | Arm | Dice (%) | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| hippocampus_003 | baseline | 84.48021022 | 2572 | 164 | 781 |
| hippocampus_003 | bands | 84.43388757 | 2666 | 296 | 687 |
| hippocampus_003 | bands_edge | 86.45424580 | 2754 | 264 | 599 |
| hippocampus_025 | baseline | 84.20716113 | 2634 | 296 | 692 |
| hippocampus_025 | bands | 84.90595368 | 2731 | 376 | 595 |
| hippocampus_025 | bands_edge | 85.48924568 | 2822 | 454 | 504 |
| hippocampus_026 | baseline | 81.59727156 | 2871 | 538 | 757 |
| hippocampus_026 | bands | 82.61661297 | 2949 | 562 | 679 |
| hippocampus_026 | bands_edge | 85.55368184 | 3044 | 444 | 584 |
| hippocampus_034 | baseline | 87.79364348 | 2859 | 279 | 516 |
| hippocampus_034 | bands | 87.16600964 | 2985 | 489 | 390 |
| hippocampus_034 | bands_edge | 91.62241888 | 3106 | 299 | 269 |
| hippocampus_036 | baseline | 82.29913097 | 2699 | 351 | 810 |
| hippocampus_036 | bands | 85.98686268 | 2749 | 136 | 760 |
| hippocampus_036 | bands_edge | 88.53008378 | 2906 | 150 | 603 |
| hippocampus_044 | baseline | 86.86868687 | 2666 | 252 | 554 |
| hippocampus_044 | bands | 87.70806658 | 2740 | 288 | 480 |
| hippocampus_044 | bands_edge | 88.32265124 | 2825 | 352 | 395 |
| hippocampus_045 | baseline | 82.52935863 | 2284 | 383 | 584 |
| hippocampus_045 | bands | 85.36628420 | 2418 | 379 | 450 |
| hippocampus_045 | bands_edge | 86.68893969 | 2465 | 354 | 403 |
| hippocampus_064 | baseline | 81.26627381 | 2965 | 672 | 695 |
| hippocampus_064 | bands | 82.72604588 | 3065 | 685 | 595 |
| hippocampus_064 | bands_edge | 83.71602006 | 3172 | 746 | 488 |
| hippocampus_087 | baseline | 86.38110928 | 3146 | 431 | 561 |
| hippocampus_087 | bands | 86.16579617 | 3217 | 543 | 490 |
| hippocampus_087 | bands_edge | 86.06203377 | 3288 | 646 | 419 |
| hippocampus_093 | baseline | 76.72990310 | 2811 | 774 | 931 |
| hippocampus_093 | bands | 78.80724175 | 2960 | 810 | 782 |
| hippocampus_093 | bands_edge | 82.07395498 | 3063 | 659 | 679 |
| hippocampus_094 | baseline | 82.08477398 | 3205 | 575 | 824 |
| hippocampus_094 | bands | 84.94881840 | 3361 | 523 | 668 |
| hippocampus_094 | bands_edge | 86.84309545 | 3462 | 482 | 567 |
| hippocampus_106 | baseline | 78.91839378 | 2437 | 647 | 655 |
| hippocampus_106 | bands | 81.19644301 | 2511 | 582 | 581 |
| hippocampus_106 | bands_edge | 82.85035629 | 2616 | 607 | 476 |
| hippocampus_114 | baseline | 84.60033538 | 3027 | 425 | 677 |
| hippocampus_114 | bands | 84.72960087 | 3110 | 527 | 594 |
| hippocampus_114 | bands_edge | 87.30482010 | 3215 | 446 | 489 |
| hippocampus_144 | baseline | 85.39795279 | 2044 | 272 | 427 |
| hippocampus_144 | bands | 83.96188964 | 2115 | 452 | 356 |
| hippocampus_144 | bands_edge | 86.05015674 | 2196 | 437 | 275 |
| hippocampus_149 | baseline | 90.01331558 | 2704 | 161 | 439 |
| hippocampus_149 | bands | 89.22040423 | 2781 | 310 | 362 |
| hippocampus_149 | bands_edge | 90.32771238 | 2839 | 304 | 304 |
| hippocampus_154 | baseline | 87.93159084 | 2725 | 216 | 532 |
| hippocampus_154 | bands | 88.97933626 | 2842 | 289 | 415 |
| hippocampus_154 | bands_edge | 89.95113012 | 2945 | 346 | 312 |
| hippocampus_155 | baseline | 80.68669528 | 2726 | 476 | 829 |
| hippocampus_155 | bands | 83.22504430 | 2818 | 399 | 737 |
| hippocampus_155 | bands_edge | 84.52622428 | 2917 | 430 | 638 |
| hippocampus_162 | baseline | 86.07704516 | 2983 | 528 | 437 |
| hippocampus_162 | bands | 84.54697054 | 3042 | 734 | 378 |
| hippocampus_162 | bands_edge | 86.63263898 | 3140 | 689 | 280 |
| hippocampus_163 | baseline | 75.25614035 | 2681 | 941 | 822 |
| hippocampus_163 | bands | 76.83692980 | 2813 | 1006 | 690 |
| hippocampus_163 | bands_edge | 79.39327685 | 2905 | 910 | 598 |
| hippocampus_166 | baseline | 85.09566969 | 2535 | 396 | 492 |
| hippocampus_166 | bands | 84.57403486 | 2596 | 516 | 431 |
| hippocampus_166 | bands_edge | 85.25896414 | 2675 | 573 | 352 |
| hippocampus_171 | baseline | 85.54963262 | 2969 | 298 | 705 |
| hippocampus_171 | bands | 86.74766883 | 3070 | 334 | 604 |
| hippocampus_171 | bands_edge | 87.83482143 | 3148 | 346 | 526 |
| hippocampus_177 | baseline | 84.62830743 | 2015 | 154 | 578 |
| hippocampus_177 | bands | 85.28577219 | 2104 | 237 | 489 |
| hippocampus_177 | bands_edge | 87.39393939 | 2163 | 194 | 430 |
| hippocampus_203 | baseline | 86.78972132 | 2398 | 301 | 429 |
| hippocampus_203 | bands | 86.10965143 | 2458 | 424 | 369 |
| hippocampus_203 | bands_edge | 86.94146172 | 2510 | 437 | 317 |
| hippocampus_215 | baseline | 85.62186641 | 2647 | 203 | 686 |
| hippocampus_215 | bands | 87.44862472 | 2766 | 227 | 567 |
| hippocampus_215 | bands_edge | 87.54658385 | 2819 | 288 | 514 |
| hippocampus_223 | baseline | 85.18463148 | 2849 | 338 | 653 |
| hippocampus_223 | bands | 86.08452363 | 2923 | 366 | 579 |
| hippocampus_223 | bands_edge | 86.57883307 | 2990 | 415 | 512 |
| hippocampus_228 | baseline | 74.75332193 | 2841 | 1089 | 830 |
| hippocampus_228 | bands | 81.26568163 | 2915 | 588 | 756 |
| hippocampus_228 | bands_edge | 81.41228191 | 2963 | 645 | 708 |
| hippocampus_231 | baseline | 81.97685821 | 2090 | 362 | 557 |
| hippocampus_231 | bands | 83.77649326 | 2174 | 369 | 473 |
| hippocampus_231 | bands_edge | 86.54333009 | 2222 | 266 | 425 |
| hippocampus_234 | baseline | 80.15279325 | 2518 | 549 | 698 |
| hippocampus_234 | bands | 81.37952856 | 2572 | 533 | 644 |
| hippocampus_234 | bands_edge | 83.80047506 | 2646 | 453 | 570 |
| hippocampus_235 | baseline | 71.82827535 | 2426 | 1289 | 614 |
| hippocampus_235 | bands | 77.46347529 | 2492 | 902 | 548 |
| hippocampus_235 | bands_edge | 79.03875969 | 2549 | 861 | 491 |
| hippocampus_236 | baseline | 75.47047372 | 2326 | 749 | 763 |
| hippocampus_236 | bands | 76.95453830 | 2446 | 822 | 643 |
| hippocampus_236 | bands_edge | 79.09693716 | 2505 | 740 | 584 |
| hippocampus_238 | baseline | 85.16627570 | 3086 | 174 | 901 |
| hippocampus_238 | bands | 86.34794157 | 3251 | 292 | 736 |
| hippocampus_238 | bands_edge | 88.32421162 | 3389 | 298 | 598 |
| hippocampus_245 | baseline | 74.63955077 | 2459 | 805 | 866 |
| hippocampus_245 | bands | 77.18120805 | 2530 | 701 | 795 |
| hippocampus_245 | bands_edge | 78.63479176 | 2615 | 711 | 710 |
| hippocampus_250 | baseline | 77.60409058 | 2656 | 758 | 775 |
| hippocampus_250 | bands | 83.19164802 | 2789 | 485 | 642 |
| hippocampus_250 | bands_edge | 85.41885209 | 2850 | 392 | 581 |
| hippocampus_263 | baseline | 79.61517011 | 2855 | 783 | 679 |
| hippocampus_263 | bands | 80.71038251 | 2954 | 832 | 580 |
| hippocampus_263 | bands_edge | 87.02355460 | 3048 | 423 | 486 |
| hippocampus_274 | baseline | 85.72555205 | 2174 | 233 | 491 |
| hippocampus_274 | bands | 85.36447542 | 2266 | 378 | 399 |
| hippocampus_274 | bands_edge | 85.70363467 | 2299 | 401 | 366 |
| hippocampus_287 | baseline | 83.79715523 | 2710 | 58 | 990 |
| hippocampus_287 | bands | 84.57212342 | 2782 | 97 | 918 |
| hippocampus_287 | bands_edge | 86.87769357 | 2923 | 106 | 777 |
| hippocampus_295 | baseline | 79.54451914 | 2899 | 762 | 729 |
| hippocampus_295 | bands | 80.57823585 | 3010 | 833 | 618 |
| hippocampus_295 | bands_edge | 83.76863120 | 3063 | 622 | 565 |
| hippocampus_299 | baseline | 82.33355307 | 2498 | 368 | 704 |
| hippocampus_299 | bands | 80.70897294 | 2550 | 567 | 652 |
| hippocampus_299 | bands_edge | 81.95535995 | 2607 | 553 | 595 |
| hippocampus_300 | baseline | 82.06236493 | 2658 | 416 | 746 |
| hippocampus_300 | bands | 85.01001387 | 2759 | 328 | 645 |
| hippocampus_300 | bands_edge | 86.09311436 | 2857 | 376 | 547 |
| hippocampus_309 | baseline | 75.55680067 | 2697 | 830 | 915 |
| hippocampus_309 | bands | 78.92122072 | 2780 | 653 | 832 |
| hippocampus_309 | bands_edge | 81.78470255 | 2887 | 561 | 725 |
| hippocampus_316 | baseline | 86.42441403 | 2747 | 515 | 348 |
| hippocampus_316 | bands | 86.21864354 | 2784 | 579 | 311 |
| hippocampus_316 | bands_edge | 84.92252682 | 2850 | 767 | 245 |
| hippocampus_321 | baseline | 84.27605834 | 2369 | 303 | 581 |
| hippocampus_321 | bands | 84.59951873 | 2461 | 407 | 489 |
| hippocampus_321 | bands_edge | 85.50601797 | 2522 | 427 | 428 |
| hippocampus_328 | baseline | 84.03492767 | 3224 | 492 | 733 |
| hippocampus_328 | bands | 85.03331625 | 3318 | 529 | 639 |
| hippocampus_328 | bands_edge | 86.72273191 | 3403 | 488 | 554 |
| hippocampus_329 | baseline | 77.04455725 | 2732 | 685 | 943 |
| hippocampus_329 | bands | 78.79473759 | 2785 | 609 | 890 |
| hippocampus_329 | bands_edge | 80.01682793 | 2853 | 603 | 822 |
| hippocampus_336 | baseline | 77.61711972 | 2013 | 581 | 580 |
| hippocampus_336 | bands | 74.31079895 | 2116 | 986 | 477 |
| hippocampus_336 | bands_edge | 78.86867231 | 2189 | 769 | 404 |
| hippocampus_337 | baseline | 83.96045918 | 2633 | 386 | 620 |
| hippocampus_337 | bands | 84.26808311 | 2697 | 451 | 556 |
| hippocampus_337 | bands_edge | 84.90653484 | 2748 | 472 | 505 |
| hippocampus_345 | baseline | 84.65441820 | 2419 | 316 | 561 |
| hippocampus_345 | bands | 85.29107819 | 2476 | 350 | 504 |
| hippocampus_345 | bands_edge | 87.16285422 | 2553 | 325 | 427 |
| hippocampus_351 | baseline | 80.53135266 | 2819 | 458 | 905 |
| hippocampus_351 | bands | 80.89425319 | 2949 | 618 | 775 |
| hippocampus_351 | bands_edge | 81.77850518 | 2998 | 610 | 726 |
| hippocampus_352 | baseline | 79.01620699 | 2779 | 576 | 900 |
| hippocampus_352 | bands | 79.72728538 | 2865 | 643 | 814 |
| hippocampus_352 | bands_edge | 81.40013727 | 2965 | 641 | 714 |
| hippocampus_354 | baseline | 78.92965332 | 2345 | 685 | 567 |
| hippocampus_354 | bands | 78.70448062 | 2345 | 702 | 567 |
| hippocampus_354 | bands_edge | 82.98017772 | 2428 | 512 | 484 |
| hippocampus_355 | baseline | 79.91097923 | 2693 | 712 | 642 |
| hippocampus_355 | bands | 81.54580981 | 2817 | 757 | 518 |
| hippocampus_355 | bands_edge | 83.33571735 | 2913 | 743 | 422 |
| hippocampus_360 | baseline | 86.28975265 | 2442 | 287 | 489 |
| hippocampus_360 | bands | 86.90517541 | 2502 | 325 | 429 |
| hippocampus_360 | bands_edge | 87.31723903 | 2568 | 383 | 363 |

### fold1_seed191

Source: `/mnt/beegfsstudents/home/3160552/medsam3_replication_20260928/results/fold1_seed191`

Original results SHA256: `f58bdb51b43bdbdabe5260aadaf9da710f2dd1d3bd2cf8d049780d9f271b6e08`

Independent run audit: **passed**. Training seed: 191; updates per arm: 200.

Support volumes: hippocampus_130, hippocampus_221, hippocampus_251, hippocampus_333, hippocampus_385.

| Case | Arm | Dice (%) | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| hippocampus_003 | baseline | 85.55788118 | 2657 | 201 | 696 |
| hippocampus_003 | bands | 85.18290808 | 2771 | 382 | 582 |
| hippocampus_003 | bands_edge | 85.98802395 | 2872 | 455 | 481 |
| hippocampus_025 | baseline | 85.69624250 | 2714 | 294 | 612 |
| hippocampus_025 | bands | 85.35012285 | 2779 | 407 | 547 |
| hippocampus_025 | bands_edge | 86.02214906 | 2874 | 482 | 452 |
| hippocampus_026 | baseline | 81.33277569 | 2917 | 628 | 711 |
| hippocampus_026 | bands | 81.24410933 | 3017 | 782 | 611 |
| hippocampus_026 | bands_edge | 81.12719752 | 3138 | 970 | 490 |
| hippocampus_034 | baseline | 89.44511460 | 2966 | 291 | 409 |
| hippocampus_034 | bands | 86.85924667 | 3067 | 620 | 308 |
| hippocampus_034 | bands_edge | 90.57918421 | 3120 | 394 | 255 |
| hippocampus_036 | baseline | 87.28905403 | 2819 | 131 | 690 |
| hippocampus_036 | bands | 88.01469463 | 2875 | 149 | 634 |
| hippocampus_036 | bands_edge | 85.75926721 | 2996 | 482 | 513 |
| hippocampus_044 | baseline | 87.42380494 | 2725 | 289 | 495 |
| hippocampus_044 | bands | 86.68310727 | 2812 | 456 | 408 |
| hippocampus_044 | bands_edge | 88.53571980 | 2869 | 392 | 351 |
| hippocampus_045 | baseline | 84.86163863 | 2346 | 315 | 522 |
| hippocampus_045 | bands | 86.52329749 | 2414 | 298 | 454 |
| hippocampus_045 | bands_edge | 81.62003904 | 2509 | 771 | 359 |
| hippocampus_064 | baseline | 82.03696317 | 3085 | 776 | 575 |
| hippocampus_064 | bands | 83.46911958 | 3176 | 774 | 484 |
| hippocampus_064 | bands_edge | 80.59103431 | 3218 | 1108 | 442 |
| hippocampus_087 | baseline | 84.53745668 | 3171 | 624 | 536 |
| hippocampus_087 | bands | 81.85106648 | 3281 | 1029 | 426 |
| hippocampus_087 | bands_edge | 85.67397717 | 3340 | 750 | 367 |
| hippocampus_093 | baseline | 81.07809261 | 2933 | 560 | 809 |
| hippocampus_093 | bands | 83.00449041 | 3050 | 557 | 692 |
| hippocampus_093 | bands_edge | 80.92470619 | 3133 | 868 | 609 |
| hippocampus_094 | baseline | 85.10583041 | 3277 | 395 | 752 |
| hippocampus_094 | bands | 85.99192327 | 3407 | 488 | 622 |
| hippocampus_094 | bands_edge | 85.62347188 | 3502 | 649 | 527 |
| hippocampus_106 | baseline | 81.94259012 | 2455 | 445 | 637 |
| hippocampus_106 | bands | 84.51305633 | 2573 | 424 | 519 |
| hippocampus_106 | bands_edge | 79.57344548 | 2649 | 917 | 443 |
| hippocampus_114 | baseline | 84.08807644 | 3036 | 481 | 668 |
| hippocampus_114 | bands | 83.45744681 | 3138 | 678 | 566 |
| hippocampus_114 | bands_edge | 85.96467574 | 3188 | 525 | 516 |
| hippocampus_144 | baseline | 85.40562576 | 2095 | 340 | 376 |
| hippocampus_144 | bands | 85.65217391 | 2167 | 422 | 304 |
| hippocampus_144 | bands_edge | 84.56273764 | 2224 | 565 | 247 |
| hippocampus_149 | baseline | 86.33669710 | 2695 | 405 | 448 |
| hippocampus_149 | bands | 83.26077166 | 2773 | 745 | 370 |
| hippocampus_149 | bands_edge | 84.96634256 | 2840 | 702 | 303 |
| hippocampus_154 | baseline | 88.15331010 | 2783 | 274 | 474 |
| hippocampus_154 | bands | 88.79950495 | 2870 | 337 | 387 |
| hippocampus_154 | bands_edge | 89.79843053 | 2918 | 324 | 339 |
| hippocampus_155 | baseline | 84.09645910 | 2755 | 242 | 800 |
| hippocampus_155 | bands | 82.90794379 | 2891 | 528 | 664 |
| hippocampus_155 | bands_edge | 84.32983848 | 2976 | 527 | 579 |
| hippocampus_162 | baseline | 86.38631504 | 3030 | 565 | 390 |
| hippocampus_162 | bands | 86.08851510 | 3122 | 711 | 298 |
| hippocampus_162 | bands_edge | 85.63736707 | 3181 | 828 | 239 |
| hippocampus_163 | baseline | 71.93603356 | 2744 | 1382 | 759 |
| hippocampus_163 | bands | 78.71003769 | 2819 | 841 | 684 |
| hippocampus_163 | bands_edge | 70.59252064 | 2907 | 1826 | 596 |
| hippocampus_166 | baseline | 83.08047493 | 2519 | 518 | 508 |
| hippocampus_166 | bands | 81.83532370 | 2604 | 733 | 423 |
| hippocampus_166 | bands_edge | 82.76390378 | 2701 | 799 | 326 |
| hippocampus_171 | baseline | 86.23878924 | 3077 | 385 | 597 |
| hippocampus_171 | bands | 86.95531502 | 3133 | 399 | 541 |
| hippocampus_171 | bands_edge | 86.11859838 | 3195 | 551 | 479 |
| hippocampus_177 | baseline | 84.94903058 | 2125 | 285 | 468 |
| hippocampus_177 | bands | 87.44744745 | 2184 | 218 | 409 |
| hippocampus_177 | bands_edge | 86.20556414 | 2231 | 352 | 362 |
| hippocampus_203 | baseline | 86.55866121 | 2431 | 359 | 396 |
| hippocampus_203 | bands | 88.02692171 | 2485 | 334 | 342 |
| hippocampus_203 | bands_edge | 82.60516485 | 2543 | 787 | 284 |
| hippocampus_215 | baseline | 88.54568855 | 2752 | 131 | 581 |
| hippocampus_215 | bands | 88.03887757 | 2808 | 238 | 525 |
| hippocampus_215 | bands_edge | 86.33669235 | 2913 | 502 | 420 |
| hippocampus_223 | baseline | 86.32236940 | 2900 | 317 | 602 |
| hippocampus_223 | bands | 86.07741190 | 2980 | 442 | 522 |
| hippocampus_223 | bands_edge | 87.09859155 | 3092 | 506 | 410 |
| hippocampus_228 | baseline | 81.48993863 | 2855 | 481 | 816 |
| hippocampus_228 | bands | 79.83181880 | 2943 | 759 | 728 |
| hippocampus_228 | bands_edge | 75.41557305 | 3017 | 1313 | 654 |
| hippocampus_231 | baseline | 81.35981665 | 2130 | 459 | 517 |
| hippocampus_231 | bands | 82.79282717 | 2170 | 425 | 477 |
| hippocampus_231 | bands_edge | 83.54668675 | 2219 | 446 | 428 |
| hippocampus_234 | baseline | 84.09350989 | 2572 | 329 | 644 |
| hippocampus_234 | bands | 83.39682540 | 2627 | 457 | 589 |
| hippocampus_234 | bands_edge | 83.27634442 | 2679 | 539 | 537 |
| hippocampus_235 | baseline | 75.31034483 | 2457 | 1028 | 583 |
| hippocampus_235 | bands | 70.60972183 | 2513 | 1565 | 527 |
| hippocampus_235 | bands_edge | 74.35521688 | 2537 | 1247 | 503 |
| hippocampus_236 | baseline | 74.89000629 | 2383 | 892 | 706 |
| hippocampus_236 | bands | 76.30645907 | 2475 | 923 | 614 |
| hippocampus_236 | bands_edge | 76.69536324 | 2539 | 993 | 550 |
| hippocampus_238 | baseline | 87.10449349 | 3276 | 259 | 711 |
| hippocampus_238 | bands | 86.12270715 | 3404 | 514 | 583 |
| hippocampus_238 | bands_edge | 85.40985591 | 3527 | 745 | 460 |
| hippocampus_245 | baseline | 75.69938001 | 2503 | 785 | 822 |
| hippocampus_245 | bands | 77.49627422 | 2600 | 785 | 725 |
| hippocampus_245 | bands_edge | 75.78225461 | 2652 | 1022 | 673 |
| hippocampus_250 | baseline | 87.36134979 | 2796 | 174 | 635 |
| hippocampus_250 | bands | 87.40831296 | 2860 | 253 | 571 |
| hippocampus_250 | bands_edge | 86.89073185 | 2986 | 456 | 445 |
| hippocampus_263 | baseline | 83.08001147 | 2897 | 543 | 637 |
| hippocampus_263 | bands | 85.63631140 | 2981 | 447 | 553 |
| hippocampus_263 | bands_edge | 81.47658111 | 3079 | 945 | 455 |
| hippocampus_274 | baseline | 84.66489966 | 2236 | 381 | 429 |
| hippocampus_274 | bands | 84.62544368 | 2265 | 423 | 400 |
| hippocampus_274 | bands_edge | 82.34465763 | 2339 | 677 | 326 |
| hippocampus_287 | baseline | 81.40643623 | 2732 | 280 | 968 |
| hippocampus_287 | bands | 83.12816799 | 2870 | 335 | 830 |
| hippocampus_287 | bands_edge | 84.76397967 | 2918 | 267 | 782 |
| hippocampus_295 | baseline | 80.37052173 | 2950 | 763 | 678 |
| hippocampus_295 | bands | 84.65564738 | 3073 | 559 | 555 |
| hippocampus_295 | bands_edge | 80.75032342 | 3121 | 981 | 507 |
| hippocampus_299 | baseline | 82.13545817 | 2577 | 496 | 625 |
| hippocampus_299 | bands | 80.18250951 | 2636 | 737 | 566 |
| hippocampus_299 | bands_edge | 80.88434351 | 2689 | 758 | 513 |
| hippocampus_300 | baseline | 85.06172840 | 2756 | 320 | 648 |
| hippocampus_300 | bands | 84.81280193 | 2809 | 411 | 595 |
| hippocampus_300 | bands_edge | 83.77445339 | 2912 | 636 | 492 |
| hippocampus_309 | baseline | 79.44737217 | 2789 | 620 | 823 |
| hippocampus_309 | bands | 80.58947368 | 2871 | 642 | 741 |
| hippocampus_309 | bands_edge | 76.16130679 | 2984 | 1240 | 628 |
| hippocampus_316 | baseline | 86.76177328 | 2828 | 596 | 267 |
| hippocampus_316 | bands | 85.23144747 | 2900 | 810 | 195 |
| hippocampus_316 | bands_edge | 83.87561186 | 2913 | 938 | 182 |
| hippocampus_321 | baseline | 84.10256410 | 2460 | 440 | 490 |
| hippocampus_321 | bands | 84.22628952 | 2531 | 529 | 419 |
| hippocampus_321 | bands_edge | 84.33141920 | 2570 | 575 | 380 |
| hippocampus_328 | baseline | 84.61244750 | 3324 | 576 | 633 |
| hippocampus_328 | bands | 81.35354247 | 3462 | 1092 | 495 |
| hippocampus_328 | bands_edge | 78.21304299 | 3484 | 1468 | 473 |
| hippocampus_329 | baseline | 78.32365850 | 2766 | 622 | 909 |
| hippocampus_329 | bands | 79.32383347 | 2839 | 644 | 836 |
| hippocampus_329 | bands_edge | 78.12500000 | 2875 | 810 | 800 |
| hippocampus_336 | baseline | 77.14497041 | 2086 | 729 | 507 |
| hippocampus_336 | bands | 72.77683135 | 2136 | 1141 | 457 |
| hippocampus_336 | bands_edge | 75.51440329 | 2202 | 1037 | 391 |
| hippocampus_337 | baseline | 85.12489927 | 2641 | 311 | 612 |
| hippocampus_337 | bands | 87.18030691 | 2727 | 276 | 526 |
| hippocampus_337 | bands_edge | 85.13739183 | 2804 | 530 | 449 |
| hippocampus_345 | baseline | 85.37095089 | 2451 | 311 | 529 |
| hippocampus_345 | bands | 87.04839809 | 2554 | 334 | 426 |
| hippocampus_345 | bands_edge | 85.81734711 | 2617 | 502 | 363 |
| hippocampus_351 | baseline | 83.22089852 | 2899 | 344 | 825 |
| hippocampus_351 | bands | 84.07017544 | 2995 | 406 | 729 |
| hippocampus_351 | bands_edge | 79.42155106 | 3103 | 987 | 621 |
| hippocampus_352 | baseline | 82.73644204 | 2830 | 332 | 849 |
| hippocampus_352 | bands | 83.60165266 | 2934 | 406 | 745 |
| hippocampus_352 | bands_edge | 80.10575017 | 3030 | 856 | 649 |
| hippocampus_354 | baseline | 81.57314799 | 2406 | 581 | 506 |
| hippocampus_354 | bands | 79.94778920 | 2450 | 767 | 462 |
| hippocampus_354 | bands_edge | 76.20481928 | 2530 | 1198 | 382 |
| hippocampus_355 | baseline | 83.17008014 | 2802 | 601 | 533 |
| hippocampus_355 | bands | 85.70576837 | 2875 | 499 | 460 |
| hippocampus_355 | bands_edge | 80.56231414 | 2980 | 1083 | 355 |
| hippocampus_360 | baseline | 86.43949930 | 2486 | 335 | 445 |
| hippocampus_360 | bands | 86.86730506 | 2540 | 377 | 391 |
| hippocampus_360 | bands_edge | 86.56816655 | 2578 | 447 | 353 |

### fold2_seed17

Source: `/mnt/beegfsstudents/home/3160552/medsam3_replication_20260928/results/fold2_seed17`

Original results SHA256: `ff8741b15c6bd636f3412e18d609bdce735b7604cb52b6506286107b33eac676`

Independent run audit: **passed**. Training seed: 17; updates per arm: 200.

Support volumes: hippocampus_053, hippocampus_075, hippocampus_095, hippocampus_102, hippocampus_173.

| Case | Arm | Dice (%) | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| hippocampus_004 | baseline | 87.15953307 | 3136 | 362 | 562 |
| hippocampus_004 | bands | 85.67574858 | 3176 | 540 | 522 |
| hippocampus_004 | bands_edge | 82.85350318 | 3252 | 900 | 446 |
| hippocampus_007 | baseline | 90.24390244 | 2960 | 228 | 412 |
| hippocampus_007 | bands | 88.99082569 | 3007 | 379 | 365 |
| hippocampus_007 | bands_edge | 89.68817984 | 3092 | 431 | 280 |
| hippocampus_015 | baseline | 64.31173768 | 2030 | 1464 | 789 |
| hippocampus_015 | bands | 57.99245283 | 1921 | 1885 | 898 |
| hippocampus_015 | bands_edge | 66.49654758 | 2215 | 1628 | 604 |
| hippocampus_024 | baseline | 84.09961686 | 3512 | 810 | 518 |
| hippocampus_024 | bands | 85.24355301 | 3570 | 776 | 460 |
| hippocampus_024 | bands_edge | 83.09521088 | 3635 | 1084 | 395 |
| hippocampus_041 | baseline | 76.92727521 | 2819 | 747 | 944 |
| hippocampus_041 | bands | 75.61491669 | 2859 | 940 | 904 |
| hippocampus_041 | bands_edge | 75.06078055 | 2933 | 1119 | 830 |
| hippocampus_042 | baseline | 78.94295302 | 2823 | 482 | 1024 |
| hippocampus_042 | bands | 78.91343120 | 2876 | 566 | 971 |
| hippocampus_042 | bands_edge | 80.46348693 | 2986 | 589 | 861 |
| hippocampus_046 | baseline | 78.64551556 | 2578 | 686 | 714 |
| hippocampus_046 | bands | 75.81268796 | 2647 | 1044 | 645 |
| hippocampus_046 | bands_edge | 76.88619377 | 2726 | 1073 | 566 |
| hippocampus_068 | baseline | 78.32210655 | 2558 | 974 | 442 |
| hippocampus_068 | bands | 78.62733293 | 2612 | 1032 | 388 |
| hippocampus_068 | bands_edge | 78.83147387 | 2685 | 1127 | 315 |
| hippocampus_074 | baseline | 83.30658106 | 2595 | 635 | 405 |
| hippocampus_074 | bands | 81.53514180 | 2645 | 843 | 355 |
| hippocampus_074 | bands_edge | 81.10861423 | 2707 | 968 | 293 |
| hippocampus_084 | baseline | 85.93342037 | 2633 | 345 | 517 |
| hippocampus_084 | bands | 85.39147040 | 2683 | 451 | 467 |
| hippocampus_084 | bands_edge | 86.09146629 | 2739 | 474 | 411 |
| hippocampus_089 | baseline | 80.35876771 | 3091 | 916 | 595 |
| hippocampus_089 | bands | 74.43555974 | 3165 | 1653 | 521 |
| hippocampus_089 | bands_edge | 81.88660311 | 3264 | 1022 | 422 |
| hippocampus_098 | baseline | 85.49919340 | 2385 | 304 | 505 |
| hippocampus_098 | bands | 86.88642064 | 2425 | 267 | 465 |
| hippocampus_098 | bands_edge | 85.22513268 | 2489 | 462 | 401 |
| hippocampus_099 | baseline | 74.28776205 | 2073 | 973 | 462 |
| hippocampus_099 | bands | 76.65665312 | 2169 | 955 | 366 |
| hippocampus_099 | bands_edge | 70.69689337 | 2105 | 1315 | 430 |
| hippocampus_133 | baseline | 82.06174351 | 3017 | 927 | 392 |
| hippocampus_133 | bands | 80.89740226 | 3083 | 1130 | 326 |
| hippocampus_133 | bands_edge | 82.25764306 | 3148 | 1097 | 261 |
| hippocampus_135 | baseline | 85.67727609 | 2315 | 460 | 314 |
| hippocampus_135 | bands | 87.73303505 | 2353 | 382 | 276 |
| hippocampus_135 | bands_edge | 85.90844063 | 2402 | 561 | 227 |
| hippocampus_143 | baseline | 85.49682875 | 2022 | 311 | 375 |
| hippocampus_143 | bands | 79.89841766 | 2045 | 677 | 352 |
| hippocampus_143 | bands_edge | 81.87997669 | 2108 | 644 | 289 |
| hippocampus_145 | baseline | 78.26305221 | 3118 | 1314 | 418 |
| hippocampus_145 | bands | 75.55294118 | 3211 | 1753 | 325 |
| hippocampus_145 | bands_edge | 77.74094246 | 3283 | 1627 | 253 |
| hippocampus_150 | baseline | 73.90501319 | 2801 | 1691 | 287 |
| hippocampus_150 | bands | 74.85210990 | 2847 | 1672 | 241 |
| hippocampus_150 | bands_edge | 71.85524848 | 2899 | 2082 | 189 |
| hippocampus_158 | baseline | 79.17691213 | 3059 | 1256 | 353 |
| hippocampus_158 | bands | 76.23483270 | 3110 | 1637 | 302 |
| hippocampus_158 | bands_edge | 76.44498186 | 3161 | 1697 | 251 |
| hippocampus_161 | baseline | 89.40758941 | 3275 | 334 | 442 |
| hippocampus_161 | bands | 89.07945866 | 3324 | 422 | 393 |
| hippocampus_161 | bands_edge | 88.44507846 | 3410 | 584 | 307 |
| hippocampus_165 | baseline | 80.00606336 | 2639 | 935 | 384 |
| hippocampus_165 | bands | 82.14393255 | 2728 | 891 | 295 |
| hippocampus_165 | bands_edge | 75.30422334 | 2630 | 1332 | 393 |
| hippocampus_174 | baseline | 89.64036500 | 3340 | 165 | 607 |
| hippocampus_174 | bands | 89.32599003 | 3406 | 273 | 541 |
| hippocampus_174 | bands_edge | 90.21584594 | 3490 | 300 | 457 |
| hippocampus_176 | baseline | 80.33689666 | 2480 | 774 | 440 |
| hippocampus_176 | bands | 80.68451909 | 2546 | 845 | 374 |
| hippocampus_176 | bands_edge | 79.80340961 | 2598 | 993 | 322 |
| hippocampus_199 | baseline | 63.66492147 | 1520 | 685 | 1050 |
| hippocampus_199 | bands | 64.80310141 | 1588 | 743 | 982 |
| hippocampus_199 | bands_edge | 65.44561934 | 1733 | 993 | 837 |
| hippocampus_220 | baseline | 84.22433585 | 2568 | 600 | 362 |
| hippocampus_220 | bands | 77.86032689 | 2620 | 1180 | 310 |
| hippocampus_220 | bands_edge | 79.96987952 | 2655 | 1055 | 275 |
| hippocampus_224 | baseline | 77.32494860 | 3197 | 1252 | 623 |
| hippocampus_224 | bands | 79.37046005 | 3278 | 1162 | 542 |
| hippocampus_224 | bands_edge | 78.63227915 | 3369 | 1380 | 451 |
| hippocampus_225 | baseline | 73.76991150 | 2084 | 1091 | 391 |
| hippocampus_225 | bands | 86.81318681 | 2133 | 306 | 342 |
| hippocampus_225 | bands_edge | 75.95287595 | 2192 | 1105 | 283 |
| hippocampus_232 | baseline | 82.56691833 | 2406 | 559 | 457 |
| hippocampus_232 | bands | 83.24720787 | 2497 | 639 | 366 |
| hippocampus_232 | bands_edge | 82.45098039 | 2523 | 734 | 340 |
| hippocampus_252 | baseline | 79.87572377 | 2828 | 678 | 747 |
| hippocampus_252 | bands | 77.87256732 | 2921 | 1006 | 654 |
| hippocampus_252 | bands_edge | 79.84780541 | 2938 | 846 | 637 |
| hippocampus_257 | baseline | 84.63546198 | 2432 | 452 | 431 |
| hippocampus_257 | bands | 83.46004393 | 2470 | 586 | 393 |
| hippocampus_257 | bands_edge | 84.52640402 | 2521 | 581 | 342 |
| hippocampus_259 | baseline | 74.35219461 | 2109 | 644 | 811 |
| hippocampus_259 | bands | 76.78847144 | 2238 | 671 | 682 |
| hippocampus_259 | bands_edge | 75.83707283 | 2197 | 677 | 723 |
| hippocampus_265 | baseline | 85.71428571 | 2592 | 347 | 517 |
| hippocampus_265 | bands | 84.35157122 | 2644 | 516 | 465 |
| hippocampus_265 | bands_edge | 84.59971800 | 2700 | 574 | 409 |
| hippocampus_276 | baseline | 82.46638598 | 3036 | 684 | 607 |
| hippocampus_276 | bands | 80.98271156 | 3115 | 935 | 528 |
| hippocampus_276 | bands_edge | 82.67294364 | 3161 | 843 | 482 |
| hippocampus_279 | baseline | 77.89148484 | 2017 | 780 | 365 |
| hippocampus_279 | bands | 79.87604106 | 2062 | 719 | 320 |
| hippocampus_279 | bands_edge | 75.90602805 | 2084 | 1025 | 298 |
| hippocampus_297 | baseline | 79.77528090 | 2343 | 745 | 443 |
| hippocampus_297 | bands | 76.88649334 | 2425 | 1097 | 361 |
| hippocampus_297 | bands_edge | 79.90330379 | 2479 | 940 | 307 |
| hippocampus_302 | baseline | 86.93149108 | 3020 | 72 | 836 |
| hippocampus_302 | bands | 87.87410426 | 3127 | 134 | 729 |
| hippocampus_302 | bands_edge | 88.58985142 | 3160 | 118 | 696 |
| hippocampus_304 | baseline | 82.58761601 | 2981 | 702 | 555 |
| hippocampus_304 | bands | 79.10351082 | 3053 | 1130 | 483 |
| hippocampus_304 | bands_edge | 81.33682831 | 3103 | 991 | 433 |
| hippocampus_311 | baseline | 78.82903274 | 2612 | 814 | 589 |
| hippocampus_311 | bands | 76.52248150 | 2689 | 1138 | 512 |
| hippocampus_311 | bands_edge | 76.81099656 | 2794 | 1280 | 407 |
| hippocampus_320 | baseline | 80.18570102 | 2159 | 775 | 292 |
| hippocampus_320 | bands | 73.69308600 | 2185 | 1294 | 266 |
| hippocampus_320 | bands_edge | 79.39806521 | 2216 | 915 | 235 |
| hippocampus_325 | baseline | 80.31170096 | 3298 | 1081 | 536 |
| hippocampus_325 | bands | 78.76230661 | 3360 | 1338 | 474 |
| hippocampus_325 | bands_edge | 81.92567568 | 3395 | 1059 | 439 |
| hippocampus_343 | baseline | 85.16550879 | 2084 | 192 | 534 |
| hippocampus_343 | bands | 82.61806780 | 2108 | 377 | 510 |
| hippocampus_343 | bands_edge | 84.42133641 | 2192 | 383 | 426 |
| hippocampus_366 | baseline | 84.80981595 | 3456 | 715 | 523 |
| hippocampus_366 | bands | 81.63312784 | 3509 | 1109 | 470 |
| hippocampus_366 | bands_edge | 84.85923823 | 3587 | 888 | 392 |
| hippocampus_372 | baseline | 85.12791678 | 3028 | 603 | 455 |
| hippocampus_372 | bands | 83.05603620 | 3120 | 910 | 363 |
| hippocampus_372 | bands_edge | 84.63590291 | 3173 | 842 | 310 |
| hippocampus_373 | baseline | 88.08437856 | 3090 | 237 | 599 |
| hippocampus_373 | bands | 88.25238688 | 3189 | 349 | 500 |
| hippocampus_373 | bands_edge | 88.99107384 | 3290 | 415 | 399 |
| hippocampus_378 | baseline | 86.18063112 | 2376 | 381 | 381 |
| hippocampus_378 | bands | 84.40463646 | 2403 | 534 | 354 |
| hippocampus_378 | bands_edge | 86.16450216 | 2488 | 530 | 269 |
| hippocampus_383 | baseline | 83.02049004 | 2897 | 662 | 523 |
| hippocampus_383 | bands | 81.81818182 | 2916 | 792 | 504 |
| hippocampus_383 | bands_edge | 82.88910371 | 3001 | 820 | 419 |
| hippocampus_387 | baseline | 78.03007962 | 2646 | 1056 | 434 |
| hippocampus_387 | bands | 77.62096774 | 2695 | 1169 | 385 |
| hippocampus_387 | bands_edge | 77.51479290 | 2751 | 1267 | 329 |

### fold2_seed83

Source: `/mnt/beegfsstudents/home/3160552/medsam3_replication_20260928/results/fold2_seed83`

Original results SHA256: `327bbca5bafa4be76ed98e8a555c81c731b3f3a1f5f2a1a33af0e4c298aabc1c`

Independent run audit: **passed**. Training seed: 83; updates per arm: 200.

Support volumes: hippocampus_053, hippocampus_075, hippocampus_095, hippocampus_102, hippocampus_173.

| Case | Arm | Dice (%) | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| hippocampus_004 | baseline | 85.97014925 | 3168 | 504 | 530 |
| hippocampus_004 | bands | 87.53089810 | 3187 | 397 | 511 |
| hippocampus_004 | bands_edge | 87.16344846 | 3205 | 451 | 493 |
| hippocampus_007 | baseline | 90.32354703 | 3015 | 289 | 357 |
| hippocampus_007 | bands | 90.73881374 | 3052 | 303 | 320 |
| hippocampus_007 | bands_edge | 91.09700281 | 3085 | 316 | 287 |
| hippocampus_015 | baseline | 61.07576634 | 1584 | 784 | 1235 |
| hippocampus_015 | bands | 64.75495308 | 1863 | 1072 | 956 |
| hippocampus_015 | bands_edge | 64.31107354 | 1902 | 1194 | 917 |
| hippocampus_024 | baseline | 87.52009895 | 3538 | 517 | 492 |
| hippocampus_024 | bands | 86.07927412 | 3605 | 741 | 425 |
| hippocampus_024 | bands_edge | 85.44920728 | 3638 | 847 | 392 |
| hippocampus_041 | baseline | 78.70216306 | 2838 | 611 | 925 |
| hippocampus_041 | bands | 78.12414547 | 2857 | 694 | 906 |
| hippocampus_041 | bands_edge | 79.98902004 | 2914 | 609 | 849 |
| hippocampus_042 | baseline | 81.27090301 | 2916 | 413 | 931 |
| hippocampus_042 | bands | 81.44971642 | 2944 | 438 | 903 |
| hippocampus_042 | bands_edge | 81.97793216 | 3009 | 485 | 838 |
| hippocampus_046 | baseline | 80.29796291 | 2641 | 645 | 651 |
| hippocampus_046 | bands | 80.38536806 | 2670 | 681 | 622 |
| hippocampus_046 | bands_edge | 80.88147707 | 2716 | 708 | 576 |
| hippocampus_068 | baseline | 82.54829954 | 2585 | 678 | 415 |
| hippocampus_068 | bands | 81.64349320 | 2613 | 788 | 387 |
| hippocampus_068 | bands_edge | 80.14375562 | 2676 | 1002 | 324 |
| hippocampus_074 | baseline | 86.90297887 | 2611 | 398 | 389 |
| hippocampus_074 | bands | 87.48758689 | 2643 | 399 | 357 |
| hippocampus_074 | bands_edge | 88.77449352 | 2673 | 349 | 327 |
| hippocampus_084 | baseline | 87.79117793 | 2657 | 246 | 493 |
| hippocampus_084 | bands | 87.42746615 | 2712 | 342 | 438 |
| hippocampus_084 | bands_edge | 87.82454443 | 2723 | 328 | 427 |
| hippocampus_089 | baseline | 87.27272727 | 3120 | 344 | 566 |
| hippocampus_089 | bands | 85.23444848 | 3172 | 585 | 514 |
| hippocampus_089 | bands_edge | 86.58684246 | 3218 | 529 | 468 |
| hippocampus_098 | baseline | 88.08834178 | 2433 | 201 | 457 |
| hippocampus_098 | bands | 85.52516983 | 2455 | 396 | 435 |
| hippocampus_098 | bands_edge | 87.72235787 | 2515 | 329 | 375 |
| hippocampus_099 | baseline | 74.40633245 | 1974 | 797 | 561 |
| hippocampus_099 | bands | 75.62995595 | 2146 | 994 | 389 |
| hippocampus_099 | bands_edge | 73.75550288 | 2178 | 1193 | 357 |
| hippocampus_133 | baseline | 87.06945229 | 3084 | 591 | 325 |
| hippocampus_133 | bands | 83.94389214 | 3082 | 852 | 327 |
| hippocampus_133 | bands_edge | 86.15426547 | 3161 | 768 | 248 |
| hippocampus_135 | baseline | 89.86630498 | 2319 | 213 | 310 |
| hippocampus_135 | bands | 89.67380815 | 2323 | 229 | 306 |
| hippocampus_135 | bands_edge | 90.18638266 | 2371 | 258 | 258 |
| hippocampus_143 | baseline | 86.50126156 | 2057 | 302 | 340 |
| hippocampus_143 | bands | 84.77178423 | 2043 | 380 | 354 |
| hippocampus_143 | bands_edge | 80.28303691 | 2099 | 733 | 298 |
| hippocampus_145 | baseline | 80.96676737 | 3216 | 1192 | 320 |
| hippocampus_145 | bands | 76.70907145 | 3226 | 1649 | 310 |
| hippocampus_145 | bands_edge | 77.44247890 | 3349 | 1764 | 187 |
| hippocampus_150 | baseline | 82.74549978 | 2827 | 918 | 261 |
| hippocampus_150 | bands | 78.82646001 | 2848 | 1290 | 240 |
| hippocampus_150 | bands_edge | 78.58583143 | 2923 | 1428 | 165 |
| hippocampus_158 | baseline | 84.55661664 | 3099 | 819 | 313 |
| hippocampus_158 | bands | 79.26395939 | 3123 | 1345 | 289 |
| hippocampus_158 | bands_edge | 84.70714381 | 3160 | 889 | 252 |
| hippocampus_161 | baseline | 89.16890080 | 3326 | 417 | 391 |
| hippocampus_161 | bands | 90.94143404 | 3323 | 268 | 394 |
| hippocampus_161 | bands_edge | 91.09230149 | 3390 | 336 | 327 |
| hippocampus_165 | baseline | 86.39205001 | 2695 | 521 | 328 |
| hippocampus_165 | bands | 89.49454906 | 2709 | 322 | 314 |
| hippocampus_165 | bands_edge | 90.55932751 | 2801 | 362 | 222 |
| hippocampus_174 | baseline | 89.68716861 | 3383 | 214 | 564 |
| hippocampus_174 | bands | 89.98284281 | 3409 | 221 | 538 |
| hippocampus_174 | bands_edge | 90.77002852 | 3501 | 266 | 446 |
| hippocampus_176 | baseline | 81.78319436 | 2550 | 766 | 370 |
| hippocampus_176 | bands | 81.84556842 | 2581 | 806 | 339 |
| hippocampus_176 | bands_edge | 84.67268981 | 2671 | 718 | 249 |
| hippocampus_199 | baseline | 69.91404011 | 1708 | 608 | 862 |
| hippocampus_199 | bands | 73.54213532 | 2125 | 1084 | 445 |
| hippocampus_199 | bands_edge | 71.36379491 | 2046 | 1118 | 524 |
| hippocampus_220 | baseline | 74.16240772 | 2612 | 1502 | 318 |
| hippocampus_220 | bands | 74.14038079 | 2609 | 1499 | 321 |
| hippocampus_220 | bands_edge | 74.44120505 | 2681 | 1592 | 249 |
| hippocampus_224 | baseline | 87.53577934 | 3364 | 502 | 456 |
| hippocampus_224 | bands | 81.50816523 | 3394 | 1114 | 426 |
| hippocampus_224 | bands_edge | 86.58917635 | 3464 | 717 | 356 |
| hippocampus_225 | baseline | 86.57937807 | 2116 | 297 | 359 |
| hippocampus_225 | bands | 87.80185759 | 2127 | 243 | 348 |
| hippocampus_225 | bands_edge | 88.26612903 | 2189 | 296 | 286 |
| hippocampus_232 | baseline | 88.16855754 | 2448 | 242 | 415 |
| hippocampus_232 | bands | 84.49399657 | 2463 | 504 | 400 |
| hippocampus_232 | bands_edge | 89.80095121 | 2549 | 265 | 314 |
| hippocampus_252 | baseline | 76.04249247 | 2398 | 334 | 1177 |
| hippocampus_252 | bands | 81.66544923 | 2795 | 475 | 780 |
| hippocampus_252 | bands_edge | 79.59556242 | 2834 | 712 | 741 |
| hippocampus_257 | baseline | 85.05509642 | 2470 | 475 | 393 |
| hippocampus_257 | bands | 84.85261544 | 2490 | 516 | 373 |
| hippocampus_257 | bands_edge | 84.48333894 | 2510 | 569 | 353 |
| hippocampus_259 | baseline | 78.47773878 | 2124 | 369 | 796 |
| hippocampus_259 | bands | 81.60645516 | 2225 | 308 | 695 |
| hippocampus_259 | bands_edge | 83.93431522 | 2479 | 508 | 441 |
| hippocampus_265 | baseline | 84.83783352 | 2655 | 495 | 454 |
| hippocampus_265 | bands | 81.42000922 | 2649 | 749 | 460 |
| hippocampus_265 | bands_edge | 82.65817023 | 2724 | 758 | 385 |
| hippocampus_276 | baseline | 83.70556363 | 3167 | 757 | 476 |
| hippocampus_276 | bands | 82.55120560 | 3184 | 887 | 459 |
| hippocampus_276 | bands_edge | 83.52758353 | 3225 | 854 | 418 |
| hippocampus_279 | baseline | 81.75876411 | 2064 | 603 | 318 |
| hippocampus_279 | bands | 80.98495212 | 2072 | 663 | 310 |
| hippocampus_279 | bands_edge | 83.41472944 | 2135 | 602 | 247 |
| hippocampus_297 | baseline | 77.53219908 | 2438 | 1065 | 348 |
| hippocampus_297 | bands | 77.49683944 | 2452 | 1090 | 334 |
| hippocampus_297 | bands_edge | 79.79156489 | 2450 | 905 | 336 |
| hippocampus_302 | baseline | 87.54641531 | 3065 | 81 | 791 |
| hippocampus_302 | bands | 87.98977853 | 3099 | 89 | 757 |
| hippocampus_302 | bands_edge | 88.29056233 | 3148 | 127 | 708 |
| hippocampus_304 | baseline | 83.79501385 | 3025 | 659 | 511 |
| hippocampus_304 | bands | 84.20618557 | 3063 | 676 | 473 |
| hippocampus_304 | bands_edge | 83.83509628 | 3091 | 747 | 445 |
| hippocampus_311 | baseline | 81.71122995 | 2674 | 670 | 527 |
| hippocampus_311 | bands | 79.90681421 | 2744 | 923 | 457 |
| hippocampus_311 | bands_edge | 79.70340215 | 2741 | 936 | 460 |
| hippocampus_320 | baseline | 89.64667214 | 2182 | 235 | 269 |
| hippocampus_320 | bands | 88.76789675 | 2201 | 307 | 250 |
| hippocampus_320 | bands_edge | 85.99657990 | 2263 | 549 | 188 |
| hippocampus_325 | baseline | 85.67049808 | 3354 | 642 | 480 |
| hippocampus_325 | bands | 81.69561521 | 3363 | 1036 | 471 |
| hippocampus_325 | bands_edge | 81.48948949 | 3392 | 1099 | 442 |
| hippocampus_343 | baseline | 84.68288791 | 2123 | 273 | 495 |
| hippocampus_343 | bands | 83.84825968 | 2144 | 352 | 474 |
| hippocampus_343 | bands_edge | 82.74024738 | 2174 | 463 | 444 |
| hippocampus_366 | baseline | 90.13688116 | 3523 | 315 | 456 |
| hippocampus_366 | bands | 90.30372347 | 3553 | 337 | 426 |
| hippocampus_366 | bands_edge | 90.56697819 | 3634 | 412 | 345 |
| hippocampus_372 | baseline | 88.83219955 | 3134 | 439 | 349 |
| hippocampus_372 | bands | 88.57383492 | 3155 | 486 | 328 |
| hippocampus_372 | bands_edge | 88.88270488 | 3194 | 510 | 289 |
| hippocampus_373 | baseline | 88.51768669 | 3153 | 282 | 536 |
| hippocampus_373 | bands | 88.65000698 | 3175 | 299 | 514 |
| hippocampus_373 | bands_edge | 89.12894845 | 3259 | 365 | 430 |
| hippocampus_378 | baseline | 87.94221152 | 2374 | 268 | 383 |
| hippocampus_378 | bands | 88.24065634 | 2420 | 308 | 337 |
| hippocampus_378 | bands_edge | 88.89287637 | 2477 | 339 | 280 |
| hippocampus_383 | baseline | 86.49369904 | 2917 | 408 | 503 |
| hippocampus_383 | bands | 87.39970282 | 2941 | 369 | 479 |
| hippocampus_383 | bands_edge | 86.65620094 | 3036 | 551 | 384 |
| hippocampus_387 | baseline | 85.79617834 | 2694 | 506 | 386 |
| hippocampus_387 | bands | 78.92898719 | 2712 | 1080 | 368 |
| hippocampus_387 | bands_edge | 85.69655600 | 2762 | 604 | 318 |

### fold2_seed191

Source: `/mnt/beegfsstudents/home/3160552/medsam3_replication_20260928/results/fold2_seed191`

Original results SHA256: `8d8726a5b9e7b083134c0726979ce3bc069c41b1176bf4a6f8d899e996ad96a7`

Independent run audit: **passed**. Training seed: 191; updates per arm: 200.

Support volumes: hippocampus_053, hippocampus_075, hippocampus_095, hippocampus_102, hippocampus_173.

| Case | Arm | Dice (%) | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| hippocampus_004 | baseline | 85.74895510 | 3180 | 539 | 518 |
| hippocampus_004 | bands | 84.39697838 | 3240 | 740 | 458 |
| hippocampus_004 | bands_edge | 85.71806051 | 3244 | 627 | 454 |
| hippocampus_007 | baseline | 91.21899254 | 2997 | 202 | 375 |
| hippocampus_007 | bands | 85.80152672 | 3091 | 742 | 281 |
| hippocampus_007 | bands_edge | 89.80596583 | 3101 | 433 | 271 |
| hippocampus_015 | baseline | 68.62713241 | 2112 | 1224 | 707 |
| hippocampus_015 | bands | 65.92538405 | 2253 | 1763 | 566 |
| hippocampus_015 | bands_edge | 65.76004835 | 2176 | 1623 | 643 |
| hippocampus_024 | baseline | 86.98420473 | 3552 | 585 | 478 |
| hippocampus_024 | bands | 83.80444856 | 3617 | 985 | 413 |
| hippocampus_024 | bands_edge | 84.09456048 | 3664 | 1020 | 366 |
| hippocampus_041 | baseline | 78.55955679 | 2836 | 621 | 927 |
| hippocampus_041 | bands | 76.96569921 | 2917 | 900 | 846 |
| hippocampus_041 | bands_edge | 78.88040712 | 2945 | 759 | 818 |
| hippocampus_042 | baseline | 80.55711874 | 2863 | 398 | 984 |
| hippocampus_042 | bands | 79.72142565 | 2919 | 557 | 928 |
| hippocampus_042 | bands_edge | 80.35617917 | 2978 | 587 | 869 |
| hippocampus_046 | baseline | 77.27272727 | 2601 | 839 | 691 |
| hippocampus_046 | bands | 78.00612156 | 2676 | 893 | 616 |
| hippocampus_046 | bands_edge | 78.88824885 | 2739 | 913 | 553 |
| hippocampus_068 | baseline | 80.08023453 | 2595 | 886 | 405 |
| hippocampus_068 | bands | 77.66372705 | 2686 | 1231 | 314 |
| hippocampus_068 | bands_edge | 78.96506359 | 2701 | 1140 | 299 |
| hippocampus_074 | baseline | 88.62495818 | 2649 | 329 | 351 |
| hippocampus_074 | bands | 87.05882353 | 2701 | 504 | 299 |
| hippocampus_074 | bands_edge | 88.87804878 | 2733 | 417 | 267 |
| hippocampus_084 | baseline | 88.47310391 | 2648 | 188 | 502 |
| hippocampus_084 | bands | 87.92549775 | 2738 | 340 | 412 |
| hippocampus_084 | bands_edge | 87.55134281 | 2771 | 409 | 379 |
| hippocampus_089 | baseline | 80.69673729 | 3104 | 903 | 582 |
| hippocampus_089 | bands | 80.28592927 | 3201 | 1087 | 485 |
| hippocampus_089 | bands_edge | 82.44835889 | 3253 | 952 | 433 |
| hippocampus_098 | baseline | 87.82086292 | 2412 | 191 | 478 |
| hippocampus_098 | bands | 89.10784665 | 2487 | 205 | 403 |
| hippocampus_098 | bands_edge | 87.02499144 | 2542 | 410 | 348 |
| hippocampus_099 | baseline | 78.68321641 | 2187 | 837 | 348 |
| hippocampus_099 | bands | 74.94081840 | 2216 | 1163 | 319 |
| hippocampus_099 | bands_edge | 76.26638328 | 2153 | 958 | 382 |
| hippocampus_133 | baseline | 85.76741525 | 2998 | 584 | 411 |
| hippocampus_133 | bands | 81.39717425 | 3111 | 1124 | 298 |
| hippocampus_133 | bands_edge | 86.27342889 | 3130 | 717 | 279 |
| hippocampus_135 | baseline | 89.67804878 | 2298 | 198 | 331 |
| hippocampus_135 | bands | 88.18518519 | 2381 | 390 | 248 |
| hippocampus_135 | bands_edge | 87.94117647 | 2392 | 419 | 237 |
| hippocampus_143 | baseline | 84.99353169 | 1971 | 270 | 426 |
| hippocampus_143 | bands | 82.38602129 | 2051 | 531 | 346 |
| hippocampus_143 | bands_edge | 80.81086400 | 2053 | 631 | 344 |
| hippocampus_145 | baseline | 79.77922730 | 3180 | 1256 | 356 |
| hippocampus_145 | bands | 74.90168864 | 3238 | 1872 | 298 |
| hippocampus_145 | bands_edge | 76.01476015 | 3296 | 1840 | 240 |
| hippocampus_150 | baseline | 82.23291065 | 2821 | 952 | 267 |
| hippocampus_150 | bands | 80.27605245 | 2908 | 1249 | 180 |
| hippocampus_150 | bands_edge | 77.45189117 | 2918 | 1529 | 170 |
| hippocampus_158 | baseline | 84.07224959 | 3072 | 824 | 340 |
| hippocampus_158 | bands | 75.05978001 | 3139 | 1813 | 273 |
| hippocampus_158 | bands_edge | 76.67230138 | 3175 | 1695 | 237 |
| hippocampus_161 | baseline | 89.15662651 | 3256 | 331 | 461 |
| hippocampus_161 | bands | 88.20606695 | 3373 | 558 | 344 |
| hippocampus_161 | bands_edge | 90.52183174 | 3400 | 395 | 317 |
| hippocampus_165 | baseline | 88.00809580 | 2609 | 297 | 414 |
| hippocampus_165 | bands | 82.40433866 | 2735 | 880 | 288 |
| hippocampus_165 | bands_edge | 85.16449410 | 2744 | 677 | 279 |
| hippocampus_174 | baseline | 89.76906552 | 3343 | 158 | 604 |
| hippocampus_174 | bands | 89.46337642 | 3426 | 286 | 521 |
| hippocampus_174 | bands_edge | 90.16922878 | 3490 | 304 | 457 |
| hippocampus_176 | baseline | 86.23001547 | 2508 | 389 | 412 |
| hippocampus_176 | bands | 80.05557271 | 2593 | 965 | 327 |
| hippocampus_176 | bands_edge | 83.37221963 | 2680 | 829 | 240 |
| hippocampus_199 | baseline | 69.80167655 | 1707 | 614 | 863 |
| hippocampus_199 | bands | 66.41965679 | 1703 | 855 | 867 |
| hippocampus_199 | bands_edge | 75.68051576 | 2113 | 901 | 457 |
| hippocampus_220 | baseline | 80.26400515 | 2493 | 789 | 437 |
| hippocampus_220 | bands | 74.58104492 | 2648 | 1523 | 282 |
| hippocampus_220 | bands_edge | 75.64029596 | 2658 | 1440 | 272 |
| hippocampus_224 | baseline | 83.93822394 | 3261 | 689 | 559 |
| hippocampus_224 | bands | 82.89489935 | 3356 | 921 | 464 |
| hippocampus_224 | bands_edge | 81.10992529 | 3420 | 1193 | 400 |
| hippocampus_225 | baseline | 85.21015067 | 2149 | 420 | 326 |
| hippocampus_225 | bands | 78.06463261 | 2162 | 902 | 313 |
| hippocampus_225 | bands_edge | 85.61417170 | 2199 | 463 | 276 |
| hippocampus_232 | baseline | 83.83855881 | 2420 | 490 | 443 |
| hippocampus_232 | bands | 81.93433608 | 2533 | 787 | 330 |
| hippocampus_232 | bands_edge | 83.12375910 | 2512 | 669 | 351 |
| hippocampus_252 | baseline | 82.50104910 | 2949 | 625 | 626 |
| hippocampus_252 | bands | 79.73351099 | 2992 | 938 | 583 |
| hippocampus_252 | bands_edge | 82.77082199 | 3041 | 732 | 534 |
| hippocampus_257 | baseline | 84.08037094 | 2448 | 512 | 415 |
| hippocampus_257 | bands | 81.43790850 | 2492 | 765 | 371 |
| hippocampus_257 | bands_edge | 83.56187015 | 2529 | 661 | 334 |
| hippocampus_259 | baseline | 84.20305289 | 2372 | 342 | 548 |
| hippocampus_259 | bands | 79.69533717 | 2145 | 318 | 775 |
| hippocampus_259 | bands_edge | 79.57683342 | 2219 | 438 | 701 |
| hippocampus_265 | baseline | 85.08541393 | 2590 | 389 | 519 |
| hippocampus_265 | bands | 83.11889630 | 2681 | 661 | 428 |
| hippocampus_265 | bands_edge | 84.73151751 | 2722 | 594 | 387 |
| hippocampus_276 | baseline | 82.79308482 | 3065 | 696 | 578 |
| hippocampus_276 | bands | 82.34527687 | 3160 | 872 | 483 |
| hippocampus_276 | bands_edge | 82.83717318 | 3200 | 883 | 443 |
| hippocampus_279 | baseline | 82.23123732 | 2027 | 521 | 355 |
| hippocampus_279 | bands | 76.89744539 | 2077 | 943 | 305 |
| hippocampus_279 | bands_edge | 81.03614924 | 2096 | 695 | 286 |
| hippocampus_297 | baseline | 76.75885492 | 2373 | 1024 | 413 |
| hippocampus_297 | bands | 74.57366723 | 2427 | 1296 | 359 |
| hippocampus_297 | bands_edge | 79.70967742 | 2471 | 943 | 315 |
| hippocampus_302 | baseline | 87.29613734 | 3051 | 83 | 805 |
| hippocampus_302 | bands | 89.02234637 | 3187 | 117 | 669 |
| hippocampus_302 | bands_edge | 88.65030675 | 3179 | 137 | 677 |
| hippocampus_304 | baseline | 80.07414272 | 3024 | 993 | 512 |
| hippocampus_304 | bands | 78.58325901 | 3084 | 1229 | 452 |
| hippocampus_304 | bands_edge | 80.94435076 | 3120 | 1053 | 416 |
| hippocampus_311 | baseline | 83.56185973 | 2651 | 493 | 550 |
| hippocampus_311 | bands | 79.76895307 | 2762 | 962 | 439 |
| hippocampus_311 | bands_edge | 79.68139030 | 2751 | 953 | 450 |
| hippocampus_320 | baseline | 85.95205072 | 2169 | 427 | 282 |
| hippocampus_320 | bands | 73.67387033 | 2250 | 1407 | 201 |
| hippocampus_320 | bands_edge | 86.07690851 | 2272 | 556 | 179 |
| hippocampus_325 | baseline | 83.96414343 | 3372 | 826 | 462 |
| hippocampus_325 | bands | 81.78997613 | 3427 | 1119 | 407 |
| hippocampus_325 | bands_edge | 82.95495495 | 3453 | 1038 | 381 |
| hippocampus_343 | baseline | 83.72278810 | 2096 | 293 | 522 |
| hippocampus_343 | bands | 81.36482940 | 2170 | 546 | 448 |
| hippocampus_343 | bands_edge | 84.45645878 | 2187 | 374 | 431 |
| hippocampus_366 | baseline | 90.11158138 | 3513 | 305 | 466 |
| hippocampus_366 | bands | 85.09831793 | 3592 | 871 | 387 |
| hippocampus_366 | bands_edge | 85.59651396 | 3634 | 878 | 345 |
| hippocampus_372 | baseline | 88.32369942 | 3056 | 381 | 427 |
| hippocampus_372 | bands | 82.94164246 | 3141 | 950 | 342 |
| hippocampus_372 | bands_edge | 89.17108568 | 3195 | 488 | 288 |
| hippocampus_373 | baseline | 87.78255443 | 3165 | 357 | 524 |
| hippocampus_373 | bands | 86.02435151 | 3250 | 617 | 439 |
| hippocampus_373 | bands_edge | 88.61505492 | 3308 | 469 | 381 |
| hippocampus_378 | baseline | 87.14524207 | 2349 | 285 | 408 |
| hippocampus_378 | bands | 85.52307150 | 2428 | 493 | 329 |
| hippocampus_378 | bands_edge | 88.44238976 | 2487 | 380 | 270 |
| hippocampus_383 | baseline | 85.72720422 | 2844 | 371 | 576 |
| hippocampus_383 | bands | 86.53421634 | 2940 | 435 | 480 |
| hippocampus_383 | bands_edge | 86.92363636 | 2988 | 467 | 432 |
| hippocampus_387 | baseline | 80.21615131 | 2672 | 910 | 408 |
| hippocampus_387 | bands | 75.53470402 | 2737 | 1430 | 343 |
| hippocampus_387 | bands_edge | 82.63544950 | 2753 | 830 | 327 |

### fold3_seed17

Source: `/mnt/beegfsstudents/home/3160552/medsam3_fold3_extension_20260929/results/fold3_seed17`

Original results SHA256: `51313dc25c7efc7a5984461b91820e91a3dd97c8d588806abcaaf23729266e8f`

Independent run audit: **passed**. Training seed: 17; updates per arm: 200.

Support volumes: hippocampus_040, hippocampus_160, hippocampus_178, hippocampus_216, hippocampus_222.

| Case | Arm | Dice (%) | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| hippocampus_001 | baseline | 83.07472802 | 2329 | 330 | 619 |
| hippocampus_001 | bands | 83.50951374 | 2370 | 358 | 578 |
| hippocampus_001 | bands_edge | 85.23002421 | 2464 | 370 | 484 |
| hippocampus_006 | baseline | 85.89201294 | 3452 | 323 | 811 |
| hippocampus_006 | bands | 86.33757168 | 3463 | 296 | 800 |
| hippocampus_006 | bands_edge | 87.61765065 | 3584 | 334 | 679 |
| hippocampus_023 | baseline | 80.09432654 | 2887 | 754 | 681 |
| hippocampus_023 | bands | 82.99942759 | 2900 | 520 | 668 |
| hippocampus_023 | bands_edge | 82.64255911 | 2971 | 651 | 597 |
| hippocampus_038 | baseline | 78.96458541 | 2776 | 697 | 782 |
| hippocampus_038 | bands | 79.57050014 | 2816 | 704 | 742 |
| hippocampus_038 | bands_edge | 78.61350870 | 2869 | 872 | 689 |
| hippocampus_039 | baseline | 86.53131255 | 3026 | 310 | 632 |
| hippocampus_039 | bands | 86.67041620 | 3082 | 372 | 576 |
| hippocampus_039 | bands_edge | 83.66996700 | 3169 | 748 | 489 |
| hippocampus_051 | baseline | 86.64009704 | 2500 | 162 | 609 |
| hippocampus_051 | bands | 87.71026433 | 2555 | 162 | 554 |
| hippocampus_051 | bands_edge | 88.43853821 | 2662 | 249 | 447 |
| hippocampus_058 | baseline | 89.89014488 | 2823 | 331 | 304 |
| hippocampus_058 | bands | 90.17046360 | 2830 | 320 | 297 |
| hippocampus_058 | bands_edge | 89.66271081 | 2871 | 406 | 256 |
| hippocampus_060 | baseline | 83.26050927 | 2763 | 465 | 646 |
| hippocampus_060 | bands | 84.31169599 | 2851 | 503 | 558 |
| hippocampus_060 | bands_edge | 83.82674639 | 2874 | 574 | 535 |
| hippocampus_067 | baseline | 80.88318551 | 2143 | 345 | 668 |
| hippocampus_067 | bands | 81.00075245 | 2153 | 352 | 658 |
| hippocampus_067 | bands_edge | 81.34790528 | 2233 | 446 | 578 |
| hippocampus_077 | baseline | 83.98546674 | 3005 | 433 | 713 |
| hippocampus_077 | bands | 86.45458471 | 3003 | 226 | 715 |
| hippocampus_077 | bands_edge | 87.33538806 | 3117 | 303 | 601 |
| hippocampus_096 | baseline | 86.61392896 | 2792 | 315 | 548 |
| hippocampus_096 | bands | 85.42865798 | 2820 | 442 | 520 |
| hippocampus_096 | bands_edge | 83.97408207 | 2916 | 689 | 424 |
| hippocampus_101 | baseline | 86.99867783 | 2961 | 252 | 633 |
| hippocampus_101 | bands | 87.81195589 | 3026 | 272 | 568 |
| hippocampus_101 | bands_edge | 88.38470320 | 3097 | 317 | 497 |
| hippocampus_105 | baseline | 83.15077320 | 2581 | 467 | 579 |
| hippocampus_105 | bands | 84.49749960 | 2619 | 420 | 541 |
| hippocampus_105 | bands_edge | 84.28504011 | 2679 | 518 | 481 |
| hippocampus_109 | baseline | 88.84546491 | 2728 | 194 | 491 |
| hippocampus_109 | bands | 88.29093281 | 2707 | 206 | 512 |
| hippocampus_109 | bands_edge | 88.97776356 | 2781 | 251 | 438 |
| hippocampus_126 | baseline | 82.80394304 | 2646 | 600 | 499 |
| hippocampus_126 | bands | 85.45137236 | 2693 | 465 | 452 |
| hippocampus_126 | bands_edge | 87.76185226 | 2786 | 418 | 359 |
| hippocampus_136 | baseline | 89.47079290 | 2443 | 205 | 370 |
| hippocampus_136 | bands | 90.62442439 | 2460 | 156 | 353 |
| hippocampus_136 | bands_edge | 90.53883583 | 2512 | 224 | 301 |
| hippocampus_141 | baseline | 83.36480687 | 2428 | 683 | 286 |
| hippocampus_141 | bands | 84.19421488 | 2445 | 649 | 269 |
| hippocampus_141 | bands_edge | 84.36816423 | 2507 | 722 | 207 |
| hippocampus_142 | baseline | 87.02037819 | 2370 | 380 | 327 |
| hippocampus_142 | bands | 86.83015441 | 2390 | 418 | 307 |
| hippocampus_142 | bands_edge | 88.87687038 | 2465 | 385 | 232 |
| hippocampus_148 | baseline | 86.94381293 | 2677 | 536 | 268 |
| hippocampus_148 | bands | 87.15877887 | 2698 | 548 | 247 |
| hippocampus_148 | bands_edge | 82.01977824 | 2737 | 992 | 208 |
| hippocampus_184 | baseline | 89.40333471 | 3244 | 408 | 361 |
| hippocampus_184 | bands | 90.18161955 | 3302 | 416 | 303 |
| hippocampus_184 | bands_edge | 89.69210174 | 3350 | 515 | 255 |
| hippocampus_193 | baseline | 83.39670469 | 2303 | 512 | 405 |
| hippocampus_193 | bands | 85.25073746 | 2312 | 404 | 396 |
| hippocampus_193 | bands_edge | 84.63045414 | 2376 | 531 | 332 |
| hippocampus_195 | baseline | 88.24287555 | 3081 | 241 | 580 |
| hippocampus_195 | bands | 88.71083131 | 3100 | 228 | 561 |
| hippocampus_195 | bands_edge | 89.81236970 | 3231 | 303 | 430 |
| hippocampus_230 | baseline | 90.82236842 | 2761 | 154 | 404 |
| hippocampus_230 | bands | 90.20501139 | 2772 | 209 | 393 |
| hippocampus_230 | bands_edge | 90.92909920 | 2892 | 304 | 273 |
| hippocampus_233 | baseline | 86.32315522 | 2714 | 407 | 453 |
| hippocampus_233 | bands | 87.65333758 | 2751 | 359 | 416 |
| hippocampus_233 | bands_edge | 87.98503974 | 2823 | 427 | 344 |
| hippocampus_242 | baseline | 89.29491684 | 3812 | 343 | 571 |
| hippocampus_242 | bands | 89.76943575 | 3874 | 374 | 509 |
| hippocampus_242 | bands_edge | 89.92528866 | 3972 | 479 | 411 |
| hippocampus_243 | baseline | 85.29028050 | 2615 | 561 | 341 |
| hippocampus_243 | bands | 81.10396738 | 2586 | 835 | 370 |
| hippocampus_243 | bands_edge | 80.63829787 | 2653 | 971 | 303 |
| hippocampus_248 | baseline | 86.35531136 | 2829 | 285 | 609 |
| hippocampus_248 | bands | 86.09933374 | 2843 | 323 | 595 |
| hippocampus_248 | bands_edge | 86.47925033 | 2907 | 378 | 531 |
| hippocampus_249 | baseline | 85.70554025 | 2800 | 385 | 549 |
| hippocampus_249 | bands | 86.67169811 | 2871 | 405 | 478 |
| hippocampus_249 | bands_edge | 87.31431967 | 2939 | 444 | 410 |
| hippocampus_253 | baseline | 86.38043787 | 3038 | 173 | 785 |
| hippocampus_253 | bands | 85.51516808 | 3129 | 366 | 694 |
| hippocampus_253 | bands_edge | 86.01975910 | 3178 | 388 | 645 |
| hippocampus_268 | baseline | 83.68912608 | 2509 | 433 | 545 |
| hippocampus_268 | bands | 83.99672400 | 2564 | 487 | 490 |
| hippocampus_268 | bands_edge | 83.71062213 | 2644 | 619 | 410 |
| hippocampus_269 | baseline | 76.32746337 | 2839 | 1161 | 600 |
| hippocampus_269 | bands | 80.37825059 | 2890 | 862 | 549 |
| hippocampus_269 | bands_edge | 77.69971377 | 2986 | 1261 | 453 |
| hippocampus_282 | baseline | 80.82164724 | 2046 | 601 | 370 |
| hippocampus_282 | bands | 80.63241107 | 2040 | 604 | 376 |
| hippocampus_282 | bands_edge | 81.49553995 | 2147 | 706 | 269 |
| hippocampus_294 | baseline | 87.41292726 | 2698 | 227 | 550 |
| hippocampus_294 | bands | 85.64683664 | 2721 | 385 | 527 |
| hippocampus_294 | bands_edge | 86.53312789 | 2808 | 434 | 440 |
| hippocampus_301 | baseline | 85.27180784 | 2698 | 170 | 762 |
| hippocampus_301 | bands | 85.64683664 | 2721 | 173 | 739 |
| hippocampus_301 | bands_edge | 87.17474981 | 2831 | 204 | 629 |
| hippocampus_303 | baseline | 85.26946108 | 2848 | 538 | 446 |
| hippocampus_303 | bands | 85.93167239 | 2880 | 529 | 414 |
| hippocampus_303 | bands_edge | 86.86280221 | 2982 | 590 | 312 |
| hippocampus_310 | baseline | 79.23331495 | 2873 | 606 | 900 |
| hippocampus_310 | bands | 79.00627901 | 2894 | 659 | 879 |
| hippocampus_310 | bands_edge | 78.60828152 | 2971 | 815 | 802 |
| hippocampus_318 | baseline | 84.89529185 | 2858 | 359 | 658 |
| hippocampus_318 | bands | 85.28634361 | 2904 | 390 | 612 |
| hippocampus_318 | bands_edge | 85.92613717 | 3013 | 484 | 503 |
| hippocampus_331 | baseline | 85.67882079 | 2761 | 490 | 433 |
| hippocampus_331 | bands | 85.31168632 | 2785 | 550 | 409 |
| hippocampus_331 | bands_edge | 83.86719321 | 2867 | 776 | 327 |
| hippocampus_341 | baseline | 85.31958003 | 2316 | 525 | 272 |
| hippocampus_341 | bands | 85.89814304 | 2336 | 515 | 252 |
| hippocampus_341 | bands_edge | 85.58072636 | 2380 | 594 | 208 |
| hippocampus_359 | baseline | 83.93515107 | 2278 | 516 | 356 |
| hippocampus_359 | bands | 84.07502758 | 2286 | 518 | 348 |
| hippocampus_359 | bands_edge | 84.13618254 | 2323 | 565 | 311 |
| hippocampus_361 | baseline | 85.01261564 | 2527 | 313 | 578 |
| hippocampus_361 | bands | 80.23627075 | 2513 | 646 | 592 |
| hippocampus_361 | bands_edge | 81.46999535 | 2627 | 717 | 478 |
| hippocampus_376 | baseline | 83.79195646 | 2771 | 262 | 810 |
| hippocampus_376 | bands | 83.61599031 | 2761 | 262 | 820 |
| hippocampus_376 | bands_edge | 82.63301305 | 2881 | 511 | 700 |
| hippocampus_380 | baseline | 88.34803772 | 2904 | 235 | 531 |
| hippocampus_380 | bands | 89.02014236 | 2939 | 229 | 496 |
| hippocampus_380 | bands_edge | 89.60827790 | 3031 | 299 | 404 |
| hippocampus_386 | baseline | 89.95308203 | 2780 | 193 | 428 |
| hippocampus_386 | bands | 90.63299233 | 2835 | 213 | 373 |
| hippocampus_386 | bands_edge | 91.65617128 | 2911 | 233 | 297 |

### fold3_seed83

Source: `/mnt/beegfsstudents/home/3160552/medsam3_fold3_extension_20260929/results/fold3_seed83`

Original results SHA256: `d84d20653de49b468b79bb284373c0428baa386805f6c9de1a26d3744ea12ed6`

Independent run audit: **passed**. Training seed: 83; updates per arm: 200.

Support volumes: hippocampus_040, hippocampus_160, hippocampus_178, hippocampus_216, hippocampus_222.

| Case | Arm | Dice (%) | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| hippocampus_001 | baseline | 81.67134831 | 2326 | 422 | 622 |
| hippocampus_001 | bands | 81.09278694 | 2434 | 621 | 514 |
| hippocampus_001 | bands_edge | 83.28877005 | 2492 | 544 | 456 |
| hippocampus_006 | baseline | 86.65568369 | 3419 | 209 | 844 |
| hippocampus_006 | bands | 86.85091089 | 3504 | 302 | 759 |
| hippocampus_006 | bands_edge | 87.46796827 | 3584 | 348 | 679 |
| hippocampus_023 | baseline | 80.34258876 | 2908 | 763 | 660 |
| hippocampus_023 | bands | 81.13645677 | 2970 | 783 | 598 |
| hippocampus_023 | bands_edge | 83.68435833 | 2980 | 574 | 588 |
| hippocampus_038 | baseline | 75.66218278 | 2728 | 925 | 830 |
| hippocampus_038 | bands | 78.47202499 | 2763 | 721 | 795 |
| hippocampus_038 | bands_edge | 76.30135153 | 2851 | 1064 | 707 |
| hippocampus_039 | baseline | 83.59859725 | 3099 | 657 | 559 |
| hippocampus_039 | bands | 84.06374502 | 3165 | 707 | 493 |
| hippocampus_039 | bands_edge | 84.04820526 | 3243 | 816 | 415 |
| hippocampus_051 | baseline | 86.70240360 | 2507 | 167 | 602 |
| hippocampus_051 | bands | 86.75360329 | 2528 | 191 | 581 |
| hippocampus_051 | bands_edge | 87.98521257 | 2618 | 224 | 491 |
| hippocampus_058 | baseline | 89.22126483 | 2744 | 280 | 383 |
| hippocampus_058 | bands | 88.89592904 | 2806 | 380 | 321 |
| hippocampus_058 | bands_edge | 89.53306174 | 2857 | 398 | 270 |
| hippocampus_060 | baseline | 83.70919881 | 2821 | 510 | 588 |
| hippocampus_060 | bands | 85.56197246 | 2889 | 455 | 520 |
| hippocampus_060 | bands_edge | 85.22465438 | 2959 | 576 | 450 |
| hippocampus_067 | baseline | 78.11889973 | 2201 | 623 | 610 |
| hippocampus_067 | bands | 80.15916079 | 2216 | 502 | 595 |
| hippocampus_067 | bands_edge | 79.49627527 | 2241 | 586 | 570 |
| hippocampus_077 | baseline | 85.40218471 | 3010 | 321 | 708 |
| hippocampus_077 | bands | 86.04780434 | 3096 | 382 | 622 |
| hippocampus_077 | bands_edge | 85.71039368 | 3146 | 477 | 572 |
| hippocampus_096 | baseline | 82.31004813 | 2822 | 695 | 518 |
| hippocampus_096 | bands | 83.81536664 | 2869 | 637 | 471 |
| hippocampus_096 | bands_edge | 81.82837914 | 2918 | 874 | 422 |
| hippocampus_101 | baseline | 87.88633368 | 3000 | 233 | 594 |
| hippocampus_101 | bands | 88.21822403 | 3040 | 258 | 554 |
| hippocampus_101 | bands_edge | 87.37864078 | 3105 | 408 | 489 |
| hippocampus_105 | baseline | 84.34278351 | 2618 | 430 | 542 |
| hippocampus_105 | bands | 84.94812450 | 2661 | 444 | 499 |
| hippocampus_105 | bands_edge | 86.12908316 | 2729 | 448 | 431 |
| hippocampus_109 | baseline | 88.25167950 | 2693 | 191 | 526 |
| hippocampus_109 | bands | 88.47790507 | 2703 | 188 | 516 |
| hippocampus_109 | bands_edge | 88.90314304 | 2772 | 245 | 447 |
| hippocampus_126 | baseline | 76.94110753 | 2626 | 1055 | 519 |
| hippocampus_126 | bands | 78.37483617 | 2691 | 1031 | 454 |
| hippocampus_126 | bands_edge | 83.76534789 | 2763 | 689 | 382 |
| hippocampus_136 | baseline | 90.02775208 | 2433 | 159 | 380 |
| hippocampus_136 | bands | 90.12144281 | 2486 | 218 | 327 |
| hippocampus_136 | bands_edge | 90.37222620 | 2525 | 250 | 288 |
| hippocampus_141 | baseline | 84.73930365 | 2446 | 613 | 268 |
| hippocampus_141 | bands | 84.08163265 | 2472 | 694 | 242 |
| hippocampus_141 | bands_edge | 83.97457344 | 2510 | 754 | 204 |
| hippocampus_142 | baseline | 87.70764120 | 2376 | 345 | 321 |
| hippocampus_142 | bands | 87.82086292 | 2412 | 384 | 285 |
| hippocampus_142 | bands_edge | 87.68297037 | 2456 | 449 | 241 |
| hippocampus_148 | baseline | 82.66047951 | 2672 | 848 | 273 |
| hippocampus_148 | bands | 84.00932401 | 2703 | 787 | 242 |
| hippocampus_148 | bands_edge | 81.01001046 | 2711 | 1037 | 234 |
| hippocampus_184 | baseline | 89.41828255 | 3228 | 387 | 377 |
| hippocampus_184 | bands | 89.73832032 | 3275 | 419 | 330 |
| hippocampus_184 | bands_edge | 89.01656868 | 3331 | 548 | 274 |
| hippocampus_193 | baseline | 84.33472500 | 2323 | 478 | 385 |
| hippocampus_193 | bands | 84.68077410 | 2341 | 480 | 367 |
| hippocampus_193 | bands_edge | 82.95059151 | 2384 | 656 | 324 |
| hippocampus_195 | baseline | 89.08780904 | 3135 | 242 | 526 |
| hippocampus_195 | bands | 89.42768537 | 3172 | 261 | 489 |
| hippocampus_195 | bands_edge | 90.30109616 | 3254 | 292 | 407 |
| hippocampus_230 | baseline | 90.45725646 | 2730 | 141 | 435 |
| hippocampus_230 | bands | 91.19322806 | 2801 | 177 | 364 |
| hippocampus_230 | bands_edge | 90.84573963 | 2868 | 281 | 297 |
| hippocampus_233 | baseline | 87.21535459 | 2681 | 300 | 486 |
| hippocampus_233 | bands | 86.15042137 | 2709 | 413 | 458 |
| hippocampus_233 | bands_edge | 86.73390225 | 2795 | 483 | 372 |
| hippocampus_242 | baseline | 89.61236679 | 3826 | 330 | 557 |
| hippocampus_242 | bands | 89.70792768 | 3870 | 375 | 513 |
| hippocampus_242 | bands_edge | 89.44043321 | 3964 | 517 | 419 |
| hippocampus_243 | baseline | 85.16528253 | 2615 | 570 | 341 |
| hippocampus_243 | bands | 84.17680974 | 2628 | 660 | 328 |
| hippocampus_243 | bands_edge | 83.78720709 | 2646 | 714 | 310 |
| hippocampus_248 | baseline | 85.88994774 | 2794 | 274 | 644 |
| hippocampus_248 | bands | 86.09471932 | 2845 | 326 | 593 |
| hippocampus_248 | bands_edge | 86.51315789 | 2893 | 357 | 545 |
| hippocampus_249 | baseline | 87.21546617 | 2797 | 268 | 552 |
| hippocampus_249 | bands | 87.98771121 | 2864 | 297 | 485 |
| hippocampus_249 | bands_edge | 86.79692126 | 2932 | 475 | 417 |
| hippocampus_253 | baseline | 84.81359649 | 3094 | 379 | 729 |
| hippocampus_253 | bands | 86.32502055 | 3150 | 325 | 673 |
| hippocampus_253 | bands_edge | 87.79489977 | 3219 | 291 | 604 |
| hippocampus_268 | baseline | 83.77445339 | 2548 | 481 | 506 |
| hippocampus_268 | bands | 83.94741113 | 2586 | 521 | 468 |
| hippocampus_268 | bands_edge | 82.81868567 | 2615 | 646 | 439 |
| hippocampus_269 | baseline | 72.57436927 | 2891 | 1637 | 548 |
| hippocampus_269 | bands | 72.37431558 | 2908 | 1689 | 531 |
| hippocampus_269 | bands_edge | 73.75201288 | 2977 | 1657 | 462 |
| hippocampus_282 | baseline | 81.58454106 | 2111 | 648 | 305 |
| hippocampus_282 | bands | 81.10984417 | 2134 | 712 | 282 |
| hippocampus_282 | bands_edge | 81.29080395 | 2179 | 766 | 237 |
| hippocampus_294 | baseline | 82.07633588 | 2688 | 614 | 560 |
| hippocampus_294 | bands | 82.91350531 | 2732 | 610 | 516 |
| hippocampus_294 | bands_edge | 83.87000150 | 2800 | 629 | 448 |
| hippocampus_301 | baseline | 83.61331220 | 2638 | 212 | 822 |
| hippocampus_301 | bands | 84.74576271 | 2750 | 280 | 710 |
| hippocampus_301 | bands_edge | 85.14068441 | 2799 | 316 | 661 |
| hippocampus_303 | baseline | 82.60616638 | 2840 | 742 | 454 |
| hippocampus_303 | bands | 86.34996993 | 2872 | 486 | 422 |
| hippocampus_303 | bands_edge | 86.05972323 | 2954 | 617 | 340 |
| hippocampus_310 | baseline | 78.89757302 | 2877 | 643 | 896 |
| hippocampus_310 | bands | 80.43656207 | 2948 | 609 | 825 |
| hippocampus_310 | bands_edge | 79.65517241 | 3003 | 764 | 770 |
| hippocampus_318 | baseline | 84.86366986 | 2879 | 390 | 637 |
| hippocampus_318 | bands | 84.21052632 | 2896 | 466 | 620 |
| hippocampus_318 | bands_edge | 85.10090167 | 2973 | 498 | 543 |
| hippocampus_331 | baseline | 87.57695343 | 2774 | 367 | 420 |
| hippocampus_331 | bands | 85.72302984 | 2801 | 540 | 393 |
| hippocampus_331 | bands_edge | 84.17954378 | 2860 | 741 | 334 |
| hippocampus_341 | baseline | 86.24092348 | 2316 | 467 | 272 |
| hippocampus_341 | bands | 86.21259409 | 2348 | 511 | 240 |
| hippocampus_341 | bands_edge | 84.63706836 | 2402 | 686 | 186 |
| hippocampus_359 | baseline | 82.08154506 | 2295 | 663 | 339 |
| hippocampus_359 | bands | 83.37219731 | 2324 | 617 | 310 |
| hippocampus_359 | bands_edge | 84.97037170 | 2366 | 569 | 268 |
| hippocampus_361 | baseline | 78.87671656 | 2556 | 820 | 549 |
| hippocampus_361 | bands | 81.73747622 | 2578 | 625 | 527 |
| hippocampus_361 | bands_edge | 79.90963855 | 2653 | 882 | 452 |
| hippocampus_376 | baseline | 84.28528256 | 2789 | 248 | 792 |
| hippocampus_376 | bands | 85.31282355 | 2884 | 296 | 697 |
| hippocampus_376 | bands_edge | 84.62204270 | 2933 | 418 | 648 |
| hippocampus_380 | baseline | 88.52963818 | 2875 | 185 | 560 |
| hippocampus_380 | bands | 87.56740563 | 2923 | 318 | 512 |
| hippocampus_380 | bands_edge | 89.33709032 | 3012 | 296 | 423 |
| hippocampus_386 | baseline | 88.13883140 | 2768 | 305 | 440 |
| hippocampus_386 | bands | 88.10420590 | 2807 | 357 | 401 |
| hippocampus_386 | bands_edge | 88.25065274 | 2873 | 430 | 335 |

### fold3_seed191

Source: `/mnt/beegfsstudents/home/3160552/medsam3_fold3_extension_20260929/results/fold3_seed191`

Original results SHA256: `1f86bc3226a79782f29b4b785962a092d4d94a0795efa5c1627574d1899c89e4`

Independent run audit: **passed**. Training seed: 191; updates per arm: 200.

Support volumes: hippocampus_040, hippocampus_160, hippocampus_178, hippocampus_216, hippocampus_222.

| Case | Arm | Dice (%) | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| hippocampus_001 | baseline | 84.43233424 | 2324 | 233 | 624 |
| hippocampus_001 | bands | 83.13464236 | 2371 | 385 | 577 |
| hippocampus_001 | bands_edge | 86.45183645 | 2495 | 329 | 453 |
| hippocampus_006 | baseline | 86.54117044 | 3379 | 167 | 884 |
| hippocampus_006 | bands | 87.47474747 | 3464 | 193 | 799 |
| hippocampus_006 | bands_edge | 89.37007874 | 3632 | 233 | 631 |
| hippocampus_023 | baseline | 79.16075985 | 2792 | 694 | 776 |
| hippocampus_023 | bands | 75.31237669 | 2863 | 1172 | 705 |
| hippocampus_023 | bands_edge | 76.75917401 | 2918 | 1117 | 650 |
| hippocampus_038 | baseline | 78.12681792 | 2686 | 632 | 872 |
| hippocampus_038 | bands | 74.09508374 | 2743 | 1103 | 815 |
| hippocampus_038 | bands_edge | 79.01269000 | 2833 | 780 | 725 |
| hippocampus_039 | baseline | 82.94616015 | 3035 | 625 | 623 |
| hippocampus_039 | bands | 82.68171436 | 3077 | 708 | 581 |
| hippocampus_039 | bands_edge | 84.14230922 | 3181 | 722 | 477 |
| hippocampus_051 | baseline | 86.04977217 | 2455 | 142 | 654 |
| hippocampus_051 | bands | 87.30894728 | 2542 | 172 | 567 |
| hippocampus_051 | bands_edge | 88.66117924 | 2639 | 205 | 470 |
| hippocampus_058 | baseline | 87.13029576 | 2725 | 403 | 402 |
| hippocampus_058 | bands | 87.79342723 | 2805 | 458 | 322 |
| hippocampus_058 | bands_edge | 89.12335144 | 2872 | 446 | 255 |
| hippocampus_060 | baseline | 83.45038168 | 2733 | 408 | 676 |
| hippocampus_060 | bands | 83.00999412 | 2824 | 571 | 585 |
| hippocampus_060 | bands_edge | 83.96723230 | 2870 | 557 | 539 |
| hippocampus_067 | baseline | 79.52270978 | 2066 | 319 | 745 |
| hippocampus_067 | bands | 79.12047302 | 2141 | 460 | 670 |
| hippocampus_067 | bands_edge | 82.81921855 | 2215 | 323 | 596 |
| hippocampus_077 | baseline | 85.86387435 | 2952 | 206 | 766 |
| hippocampus_077 | bands | 84.59623894 | 3059 | 455 | 659 |
| hippocampus_077 | bands_edge | 87.59625963 | 3185 | 369 | 533 |
| hippocampus_096 | baseline | 84.73852721 | 2779 | 440 | 561 |
| hippocampus_096 | bands | 81.20235869 | 2823 | 790 | 517 |
| hippocampus_096 | bands_edge | 84.05339457 | 2928 | 699 | 412 |
| hippocampus_101 | baseline | 86.02182247 | 2917 | 271 | 677 |
| hippocampus_101 | bands | 87.76075857 | 3008 | 253 | 586 |
| hippocampus_101 | bands_edge | 88.99344169 | 3121 | 299 | 473 |
| hippocampus_105 | baseline | 82.22615409 | 2556 | 501 | 604 |
| hippocampus_105 | bands | 82.53376805 | 2658 | 623 | 502 |
| hippocampus_105 | bands_edge | 83.64083641 | 2720 | 624 | 440 |
| hippocampus_109 | baseline | 87.46288354 | 2651 | 192 | 568 |
| hippocampus_109 | bands | 88.18770227 | 2725 | 236 | 494 |
| hippocampus_109 | bands_edge | 88.50502513 | 2818 | 331 | 401 |
| hippocampus_126 | baseline | 80.40466926 | 2583 | 697 | 562 |
| hippocampus_126 | bands | 81.21342778 | 2637 | 712 | 508 |
| hippocampus_126 | bands_edge | 84.95145631 | 2800 | 647 | 345 |
| hippocampus_136 | baseline | 88.47874720 | 2373 | 178 | 440 |
| hippocampus_136 | bands | 89.91735537 | 2448 | 184 | 365 |
| hippocampus_136 | bands_edge | 90.81100521 | 2525 | 223 | 288 |
| hippocampus_141 | baseline | 84.15910677 | 2412 | 606 | 302 |
| hippocampus_141 | bands | 80.88578089 | 2429 | 863 | 285 |
| hippocampus_141 | bands_edge | 82.47729149 | 2497 | 844 | 217 |
| hippocampus_142 | baseline | 88.34698436 | 2373 | 302 | 324 |
| hippocampus_142 | bands | 88.33179724 | 2396 | 332 | 301 |
| hippocampus_142 | bands_edge | 88.69346734 | 2471 | 404 | 226 |
| hippocampus_148 | baseline | 80.36374846 | 2607 | 936 | 338 |
| hippocampus_148 | bands | 80.21132075 | 2657 | 1023 | 288 |
| hippocampus_148 | bands_edge | 81.82498130 | 2735 | 1005 | 210 |
| hippocampus_184 | baseline | 89.30976431 | 3183 | 340 | 422 |
| hippocampus_184 | bands | 88.91319207 | 3252 | 458 | 353 |
| hippocampus_184 | bands_edge | 89.15182132 | 3353 | 564 | 252 |
| hippocampus_193 | baseline | 83.38581219 | 2251 | 440 | 457 |
| hippocampus_193 | bands | 79.78271940 | 2350 | 833 | 358 |
| hippocampus_193 | bands_edge | 83.46675948 | 2398 | 640 | 310 |
| hippocampus_195 | baseline | 88.33116977 | 3062 | 210 | 599 |
| hippocampus_195 | bands | 88.69098106 | 3137 | 276 | 524 |
| hippocampus_195 | bands_edge | 90.21347380 | 3254 | 299 | 407 |
| hippocampus_230 | baseline | 89.05037159 | 2696 | 194 | 469 |
| hippocampus_230 | bands | 90.14857881 | 2791 | 236 | 374 |
| hippocampus_230 | bands_edge | 91.63997438 | 2861 | 218 | 304 |
| hippocampus_233 | baseline | 86.65364583 | 2662 | 315 | 505 |
| hippocampus_233 | bands | 86.96206567 | 2728 | 379 | 439 |
| hippocampus_233 | bands_edge | 87.34314485 | 2819 | 469 | 348 |
| hippocampus_242 | baseline | 89.85507246 | 3782 | 253 | 601 |
| hippocampus_242 | bands | 90.15036718 | 3867 | 329 | 516 |
| hippocampus_242 | bands_edge | 90.46377140 | 3989 | 447 | 394 |
| hippocampus_243 | baseline | 79.24879714 | 2553 | 934 | 403 |
| hippocampus_243 | bands | 78.24502809 | 2577 | 1054 | 379 |
| hippocampus_243 | bands_edge | 79.99405470 | 2691 | 1081 | 265 |
| hippocampus_248 | baseline | 84.12723041 | 2711 | 296 | 727 |
| hippocampus_248 | bands | 83.69092273 | 2789 | 438 | 649 |
| hippocampus_248 | bands_edge | 85.45914571 | 2871 | 410 | 567 |
| hippocampus_249 | baseline | 84.18954148 | 2745 | 427 | 604 |
| hippocampus_249 | bands | 85.41353383 | 2840 | 461 | 509 |
| hippocampus_249 | bands_edge | 85.80588750 | 2944 | 569 | 405 |
| hippocampus_253 | baseline | 85.71019883 | 2996 | 172 | 827 |
| hippocampus_253 | bands | 83.23248622 | 3095 | 519 | 728 |
| hippocampus_253 | bands_edge | 87.06803456 | 3225 | 360 | 598 |
| hippocampus_268 | baseline | 84.26532326 | 2509 | 392 | 545 |
| hippocampus_268 | bands | 83.88788723 | 2559 | 488 | 495 |
| hippocampus_268 | bands_edge | 84.04780876 | 2637 | 584 | 417 |
| hippocampus_269 | baseline | 75.90441864 | 2843 | 1209 | 596 |
| hippocampus_269 | bands | 70.13838310 | 2889 | 1910 | 550 |
| hippocampus_269 | bands_edge | 77.85322539 | 2981 | 1238 | 458 |
| hippocampus_282 | baseline | 81.39353834 | 2091 | 631 | 325 |
| hippocampus_282 | bands | 80.42528954 | 2118 | 733 | 298 |
| hippocampus_282 | bands_edge | 80.97014925 | 2170 | 774 | 246 |
| hippocampus_294 | baseline | 84.69191271 | 2639 | 345 | 609 |
| hippocampus_294 | bands | 83.40793332 | 2702 | 529 | 546 |
| hippocampus_294 | bands_edge | 87.74839618 | 2804 | 339 | 444 |
| hippocampus_301 | baseline | 83.20000000 | 2600 | 190 | 860 |
| hippocampus_301 | bands | 84.65774980 | 2690 | 205 | 770 |
| hippocampus_301 | bands_edge | 85.58710668 | 2788 | 267 | 672 |
| hippocampus_303 | baseline | 87.52147431 | 2802 | 307 | 492 |
| hippocampus_303 | bands | 85.22573871 | 2841 | 532 | 453 |
| hippocampus_303 | bands_edge | 85.76009328 | 2942 | 625 | 352 |
| hippocampus_310 | baseline | 78.69905081 | 2819 | 572 | 954 |
| hippocampus_310 | bands | 77.69531777 | 2879 | 759 | 894 |
| hippocampus_310 | bands_edge | 78.89736384 | 2948 | 752 | 825 |
| hippocampus_318 | baseline | 83.93200296 | 2839 | 410 | 677 |
| hippocampus_318 | bands | 85.16676419 | 2911 | 409 | 605 |
| hippocampus_318 | bands_edge | 85.20794626 | 2981 | 500 | 535 |
| hippocampus_331 | baseline | 83.75495276 | 2748 | 620 | 446 |
| hippocampus_331 | bands | 82.66272189 | 2794 | 772 | 400 |
| hippocampus_331 | bands_edge | 83.12027891 | 2861 | 829 | 333 |
| hippocampus_341 | baseline | 83.89322680 | 2310 | 609 | 278 |
| hippocampus_341 | bands | 84.57495015 | 2333 | 596 | 255 |
| hippocampus_341 | bands_edge | 85.76468513 | 2431 | 650 | 157 |
| hippocampus_359 | baseline | 85.40419162 | 2282 | 428 | 352 |
| hippocampus_359 | bands | 81.66250446 | 2289 | 683 | 345 |
| hippocampus_359 | bands_edge | 84.00351185 | 2392 | 669 | 242 |
| hippocampus_361 | baseline | 74.34367542 | 2492 | 1107 | 613 |
| hippocampus_361 | bands | 81.46380270 | 2560 | 620 | 545 |
| hippocampus_361 | bands_edge | 78.59281437 | 2625 | 950 | 480 |
| hippocampus_376 | baseline | 82.79270930 | 2680 | 213 | 901 |
| hippocampus_376 | bands | 82.23305272 | 2784 | 406 | 797 |
| hippocampus_376 | bands_edge | 84.06345510 | 2888 | 402 | 693 |
| hippocampus_380 | baseline | 88.07710244 | 2833 | 165 | 602 |
| hippocampus_380 | bands | 87.86074210 | 2877 | 237 | 558 |
| hippocampus_380 | bands_edge | 88.86245353 | 2988 | 302 | 447 |
| hippocampus_386 | baseline | 87.84592976 | 2714 | 257 | 494 |
| hippocampus_386 | bands | 84.29827220 | 2781 | 609 | 427 |
| hippocampus_386 | bands_edge | 89.36235912 | 2894 | 375 | 314 |

## Superseded inference-refinement pilots

# Superseded MedSAM3 inference-refinement pilots (28 September 2026)

Archived at the user's request before deleting generated experiment artifacts.
These experiments applied bands/edge as new mask-free **post-inference
refinements**, which did not answer the intended training-loss question.

Zero-shot job 673230, hippocampus_017: baseline Dice 0.0086180980;
+bands 0.0079565302; +bands+edge 0.0079634845.

Five-shot job 673282 used training volumes 011, 056, 146, 319, 363 (prefix
hippocampus_), seed 42, final LoRA after 200 AdamW updates, four random slices
per update, LR 5e-5 with 20-update warmup/cosine decay to 5e-6. The base model
remained frozen. Whole-hippocampus binary 3D Dice on separate validation volumes:

| Case | Zero-shot | Five-shot | + bands | + bands + edge |
|---|---:|---:|---:|---:|
| hippocampus_017 | 0.008618 | 0.868163 | 0.853706 | 0.855359 |
| hippocampus_019 | 0.000514 | 0.836421 | 0.839829 | 0.840205 |
| hippocampus_033 | 0.001736 | 0.881514 | 0.859003 | 0.860000 |
| hippocampus_035 | 0.003947 | 0.873124 | 0.873990 | 0.875187 |
| hippocampus_037 | 0.008629 | 0.836256 | 0.812246 | 0.812439 |
| Mean | 0.004689 | 0.859096 | 0.847755 | 0.848638 |

Both jobs completed; saved-mask Dice and grid/hash checks passed. The original
case's repeated zero-shot output was identical. Few-shot adaptation helped,
while post-inference constraints lowered mean Dice. These results do not test
supervised constraints during fine-tuning. The old predictions, trained adapter,
plots, logs, result archives and detailed run records were authorized for
removal. Original datasets, downloaded SAM3/medical weights, reusable source
code and shared runtime are retained for the corrected training experiment.


Machine-readable original results, adapter/prediction hashes, protocols, calibrations and individual audits are preserved in `RESULT_RECORDS.json`. Removing binary masks means future spatial reanalysis of those masks will no longer be possible.
