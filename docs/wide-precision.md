# Wider precision and strong-coupling solver integration

The unreleased storage, MPFR/MPC bridge, arithmetic, dense exact GEMM, batched
GEMM, and real/complex QR APIs accept every multiple of 32 from 64 through
2048 bits. Existing layouts and rounding sequences are unchanged. 1130 requested
bits round up to 1152; 1536 and 2048 bits provide about 462 and 616 decimal digits
of significand capacity. Conditioning still consumes accuracy.

Above 1024 bits, shared Metal arithmetic uses loops with unrolling disabled and
keeps add, multiply and divide helpers out of line. Composed batched complex sums
and complex arithmetic use exact-window additions of rounded operands, preserving the original
rounding sequence while avoiding a reproduced Metal compiler defect.
QR host exact-product scratch is sized from the actual limb count. Batched complex
GEMM retains packed public storage; private padding remains at 33 words for the
older widths and uses the logical limb count above 1024 bits.
The previous kernels and padding remain selected at widths through 1024 bits.
Wide norm sum trees use the same exact-window addition, with 64-lane groups
above 1920 bits to fit the device threadgroup memory.
No speedup at the larger widths or on the production solver shape is assumed.
Split-precision GEMM is unnecessary for the requested widths and is not added.

Transcendental functions retain their separately validated **64–1024-bit** limit;
their retry tables must not be indexed with a wider storage width.

## Structural device preflight and fallback diagnostics

```cpp
#include <limbforge/capabilities.hpp>
limbforge::Engine engine;
auto limits = engine.device_limits();
auto qr = limbforge::query_qr_support(limits, 1152, 3000, 3000);
if (!qr.supported) { /* explain qr.reason and choose the CPU */ }
// Single-round real GEMM (use the expanded real dimensions for a complex embedding).
auto dense = limbforge::query_gemm_support(limits, 1152, 24000, 641, 1280);
limbforge::StridedGemm shape{100, 4, 4, 4, 16, 0, 16};
auto chain = limbforge::query_batched_gemm_support(limits, 1152, true, shape);
```

`precision_supported(bits)` is a constexpr storage/algebra-width check. The shape
queries check overflow, indexing, strides and each necessary matrix buffer against
`max_buffer_bytes`; they allocate no operands and compile no shaders. They return
matrix operand, factor, matrix-workspace and largest-buffer bytes. The device's
`recommended_working_set_bytes` is advisory and shared with other applications.
Supported shapes can still fail allocation or shader compilation; use `prewarm`
for a pipeline check and retain exception handling. Data-dependent exponent bands
can still cause exact host fallback. A QR always has host panels, and thresholds
can place small updates and solves on the host.

After a dense call, inspect `Linalg::report()` (`gpu_outputs`, `fallback_outputs`,
`left_fallback`, `right_fallback` and `resident_overflow`). After a factorization,
inspect `qr.info().gpu_updates` and `.host_updates`. A structural query cannot
predict conditioning, rank, exponent spread, or measured CPU/GPU break-even time.

## QR dimensions and matrix memory

There is no single hardware-independent maximum n for a width. For real QR,
n < 46341 and m*n < 2^31. Complex QR uses doubled real embeddings, so
n < 23171 and 4*m*n < 2^31. Both require m >= n. Device buffer length and available
memory generally impose much tighter practical limits. Dense exact GPU GEMM has
K <= 65472; larger K uses the existing exact host fallback.

Let e = bits/8 + 12 bytes per real entry, nb = min(block,n) (block 0 means n),
and t = ceil(n/nb)*nb^2 (zero for n=0). Persistent factors require:

- Real QR: e*(m*n + n*n + t + n).
- Complex QR: e*(6*m*n + 6*n*n + 6*t + 2*n), including the real embeddings.

Each factor array is rounded independently to the host page size. GPU update
matrix scratch adds e*(m*n + 2*nb*n) for real and e*(2*m*n + 4*nb*n) for complex
when n > nb. The caller's input matrix, panel host arrays, exact accumulators,
right-hand sides, driver/pipeline caches and band/CRT workspaces are additional;
these formulas are **not peak process memory**. Solve scratch depends on nrhs.
Workspace pools can retain their largest allocations until `release_workspaces`.

For square n=3000 and block=32, excluding page rounding:

| Bits | One real matrix | Real factors | Complex factors with embeddings |
|---:|---:|---:|---:|
| 1024 | 1.17 GiB | 2.36 GiB | 14.16 GiB |
| 1152 | 1.31 GiB | 2.63 GiB | 15.78 GiB |
| 1536 | 1.71 GiB | 3.44 GiB | 20.63 GiB |
| 2048 | 2.25 GiB | 4.52 GiB | 27.10 GiB |

Use `query_qr_support` for exact page-rounded factor sizes and individual buffer
checks on the actual device. Production-shape timings and memory peaks require
measurement with the client workload.

## Mixed-precision refinement with reusable factors

Both `QRFactor` and `ComplexQRFactor` already own reusable page-backed factors
accessible to Metal without re-uploading the input matrix. The producing `Linalg`
must outlive the factor. Repeated `solve(rhs,nrhs,correction)` calls reuse V, R,
tau and T; they do not refactor A or mutate the factor. Real Buffer operands are
also supported. GPU/host placement follows the factor's original options.

Factor a rounded copy of A once, for example at 1024 bits, then keep A, b and x
at the caller's full MPFR/MPC precision:

1. Compute r = b - A*x at full precision, and test the caller's stopping criterion.
2. Pack r at the factor width and call `factor.solve(r,nrhs,delta)`.
3. Unpack delta into full precision and update x = x + delta there.
4. Repeat with an iteration limit and a stagnation/rank/status check.

Do not form the residual at the factor width, or store x in a factor-width buffer.
There is no guaranteed iteration count: the low-precision factor must resolve the
conditioning. Large condition numbers can prevent convergence entirely. Set
`solve_host_macs=0` when deliberately testing device updates with few right-hand
sides; the default may correctly select the faster host exact dots.

## Client rebuild

The library now resides at `/Users/k0959535/Dropbox/Programming/2026 WolfNum`.
The old `2026 4d SoV/LimbForge` path is a compatibility symlink. Headers and the
static archive must be rebuilt together; the unreleased changes do not rename the
`limbforge` namespace, CMake package or GitHub repository.

The applied client change is preserved in [strong-coupling-client.patch](strong-coupling-client.patch).
From the Strong Coupling project root, apply it with `git apply` (or `patch -p1`)
when updating another checkout.

The client `sc_gpu.hpp` width switch has been extended through 2048 bits when
`capabilities.hpp` is available; older WolfNum releases retain the original width
switch. Each client path performs structural preflight before packing, and an
unsupported width now reports a CPU fallback reason. The client's default `SC_MAXLIMBS=24` stores at most
1536 bits per MPFR value, so a 2048-bit client run also needs `SC_MAXLIMBS>=32`.
Select the run's working precision deliberately and keep CPU fallback diagnostics.

## Tiny-value decision audit

Real QR pivot norms (`compare_magnitude`), complex QR pivot norms
(`compare_values`) and rank tests compare stored signs, exponents and limbs.
`Numerics::norm_max` compares magnitudes on the device without conversion to
binary64. The doubles in these host paths describe timings, operation counts or
CRT bit budgets, not matrix-entry magnitudes. Tests include entries around
2^-1400 and client diagonal pivots of 1e-400. No binary64 underflow fix is needed
for these decisions.


## Verification

The precision tests compare arithmetic, exact GEMM, composed/fused cached complex
GEMM, casts and norms against MPFR/MPC. QR tests compare factors, repeated solves
and Q applications to an independent MPFR replay, including pivoting, deficient
matrices and tiny entries. Run the additional width sweeps after building with
`LIMBFORGE_ENABLE_GPU_TESTS=ON`:

```sh
./build/test_wolfnum_wide --all-widths
./build/test_limbforge_qr --wide-all
./build/test_limbforge_qr_complex --wide-all
MTL_SHADER_VALIDATION=1 ./build/test_wolfnum_wide
```

The default wide test covers 1056, 1152, 1536 and 2048 bits. It also checks
full-precision residual refinement with a reused 256-bit factor, reaching an
error below 2^-1130 in eight solves. The 2048-bit case checks multi-block
`norm_max` and `norm2` on values around 2^-1400. The actual strong-coupling adapter
has been checked at 1152, 1536 and 2048 bits on small GEMM, repeated cached
batches and a real QR solve with diagonal entries of 1e-400. These correctness
checks establish no production-size timing or peak-memory result.
