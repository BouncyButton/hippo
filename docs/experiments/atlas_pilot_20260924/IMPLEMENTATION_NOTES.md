# Implementation notes, recorded before outcome scoring

- SimpleITK 2.5.2 is installed only under the ignored experiment dependency
  directory; the shared project environment was not changed.
- A one-target image-only smoke test exposed SimpleITK's copy-on-add transform
  semantics: a composite built before residual optimization retained the old
  residual. Composition was moved after optimization and an independent
  transform-order/copy test added. Superseded smoke priors and metadata are in
  `experiments/atlas_pilot_20260924/smoke_before_composition_fix/` and are excluded
  from all results. No target label scores had been computed.
- Registration runs two independent target jobs at a time, each with one ITK
  thread. This changes throughput only; the settings in PROTOCOL.md are fixed.
- The canonical checkpoint inventory records a cluster-serialized dataset pickle
  hash. The local pickle has a different serialization hash, exactly matching the
  existing CPU context/graph cache manifests. The exporter therefore checks the
  checkpoint against the canonical inventory, the local pickle against the CPU
  manifest, and each new hard prediction/reference against the prior CPU cache
  and native label. Both descriptor hashes are retained. No mismatch was waived
  for actual predictions or labels.
- Local graph inference uses the native image grid, with no source/sink nodes
  outside that grid. Pairwise regularization is applied once per shared face.
- The relative-volume-error mean is signed; improved/worsened counts compare
  absolute volume error, avoiding the false implication that greater negative
  bias is better.
