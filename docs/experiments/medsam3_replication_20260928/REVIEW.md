# Replication review

Scoped review covered the changed training entrypoint, study generation,
crossed fold/seed analysis, integrity audits, restart behavior and quota budget.
An independent adversarial reviewer found no unresolved findings after fixes.

Fixed before submission:

- Bootstrap resamples a common seed-index vector across folds, preserving the
  crossed design; case draws remain paired across selected seeds and arms.
- Primary averaging is explicitly equal-fold/equal-seed, avoiding accidental
  weighting by unequal validation-fold sizes.
- The incremental edge comparison is explicitly descriptive.
- Zero constraint-gradient calibration samples follow the existing skip policy;
  audits require at least five finite nonzero calibration samples per constraint.
- Audits bind their result to the report's SHA256. Aggregation rejects changed,
  missing or unaudited results and cannot silently drop failed runs.
- Resume behavior rejects partial directories and requires preserving failed
  attempts before a same-protocol retry.

Validation: 17 existing loss/training/inference checks plus nine new split,
bootstrap and aggregation checks passed locally; changed source compiles.
The common-seed regression catches falsely treating nine cells as nine
independent optimization seeds. Exact three-fold sign-flip tests and the
zero-effect case return the expected minimum p=0.25 and p=1, respectively.

The optional CLI cross-check was unavailable: the installed Codex launcher points
to a missing platform executable (ENOENT). The independent agent review and
local checks completed successfully.

There are no patient identifiers verified for grouping. All inference about
robustness is explicitly volume-level and exploratory with only three fold
blocks; a narrow bootstrap interval does not establish patient-level significance.

All original 26 checks also passed on the cluster. A subsequent quota audit identified two-copy BeeGFS buddy mirroring: the corrected requirement is 6.2316 GiB including reserve. An added regression test checks both copies and the remaining-run budget; the earlier 3.6158 GiB preflight bound is superseded. The worker rechecks each run's data hashes before and after execution.
