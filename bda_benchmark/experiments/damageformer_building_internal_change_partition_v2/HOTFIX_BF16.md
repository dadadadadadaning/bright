# BF16 scatter moment hotfix

Fixes CUDA AMP/BF16 failure in `association._scatter_moments`:
`RuntimeError: scatter(): Expected self.dtype to be equal to src.dtype`.

Changes:
- first/second moment scatter accumulation is explicit FP32;
- mean/std are cast back to the incoming feature dtype;
- region mass remains FP32;
- soft-peak scatter accumulation is also FP32;
- soft unpool explicitly casts add source to destination dtype;
- adds a BF16 contract test.

This changes numerical implementation only; V2 architecture and loss definitions are unchanged.
