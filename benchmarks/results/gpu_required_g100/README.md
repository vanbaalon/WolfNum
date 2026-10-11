# GPU-required g100 reconstruction check

Case: g=100, P640/Q540, WP270, alphaP=2.0, alphaQ=2.2. Input MPFR precision
is 1128 bits; GPU packing uses 1152 bits. `capture_case.cpp` regenerates the
actual production basis with the same adaptive measures as `sc_run`, both
converging at level 13. `metadata.json` records the exact basis fixture format
and SHA-256. This fixture contains numerical basis coefficients, not pointers
or native MPC objects.

`replay_case.cpp` reconstructs the actual node/basis/kernel operands used by
`build_model`, then calls the patched client adapter for both full-size GEMMs.
It excludes unrelated principal-value integrals and Newton iterations.
Both products report zero host fallback and all 28 million real-component
outputs on Metal. Forty component samples per product match independent exact
MPFR, including tile boundaries and the last row. `--default-routing` shows
that the original band limits reject every right-side column before any host
product is computed.

To replay from the WolfNum checkout with the patched client (set `client_cpp`
to the strong-coupling cpp directory):

```sh
mkdir -p work/gpu-required
cp benchmarks/results/gpu_required_g100/g100_basis.bin work/gpu-required/
clang++ -std=c++20 -O0 -DSC_LIMBFORGE -DQSC_INLINE_LIMBS -DQSC_MAXLIMBS=24 \
  -I"$client_cpp/include" -Iinclude -Itests -I/opt/homebrew/include \
  benchmarks/results/gpu_required_g100/replay_case.cpp build/liblimbforge.a \
  -L/opt/homebrew/lib -lmpc -lmpfr -lgmp -framework Metal -framework Foundation \
  -pthread -o work/gpu-required/replay_case
SC_GPU=1 SC_GPU_REQUIRE_GEMM=1 SC_GPU_PROFILE=1 work/gpu-required/replay_case
```

Compile `capture_case.cpp` with the same include/library options (no archive or
frameworks required), then run from the checkout to regenerate the fixture.
The profile includes actual device time and counters; wall times include any
first-use compilation. Other CPU solver jobs were active during validation.
These records establish GPU placement and sampled exactness, not an idle
CPU/GPU comparison, a full-solver result, or a change to QR/LU performance.

The normal/Metal API gates cover 512, 928, 1024, 1152, 1536 and 2048 bits.
Focused version/package, CPU/GPU GEMM, resident GEMM and real QR regressions
pass; the complex QR quick suite separately checks 528 factorizations/solves
and 1056 Q applications against MPFR. Startup checks reject missing/conflicting
GPU flags before precomputation.
