"""tests — lightweight guards that run on CPU without datasets.

Present (run in CI without torch via the conftest stub):
  test_manifest.py        Manifest builder + IO (stdlib only; always runs).
  test_dataset_logic.py   Loader wiring, FF++ split application, DF40 grouping,
                          balanced sampling, get_dataloaders (incl. the in-domain
                          test flag) — run under the conftest torch stub or real torch.
  test_dataset_numeric.py Numeric checks on real tensors (FFT magnitude, batch
                          shapes); marked needs_real_torch, skipped under the stub.
  test_prefusion_hook.py  Guards the BINDING INVARIANT on the ASSEMBLED detector:
                          the frequency-saliency target is pre-fusion (structural
                          containment + forward-order probe + activation identity).
                          Builds the real model -> needs_real_torch, skipped under stub.
  conftest.py             Stubs torch/torchvision/PIL when real torch is absent and
                          auto-skips needs_real_torch tests.

Planned (land with later code):
  test_shapes.py          Forward-pass shape checks for each branch, the fusion
                          module, and the assembled detector on dummy tensors.

These run in CI / pre-push with random weights and no data, so the invariants
cannot regress silently.
"""
