# WolfNum

**Multiprecision arithmetic on the Apple Silicon GPU.**

[![CI](https://github.com/vanbaalon/WolfNum/actions/workflows/ci.yml/badge.svg)](https://github.com/vanbaalon/WolfNum/actions/workflows/ci.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://en.cppreference.com/w/cpp/17)
[![Backend: Metal](https://img.shields.io/badge/backend-Metal-555555.svg)](https://developer.apple.com/metal/)

WolfNum is the standalone numerical library in the **Wolfbook family**, providing
batched, high-precision real and complex arithmetic. It stores significands in
32-bit integer limbs and executes
precision-specialized Metal kernels, supporting **64–2048 significand bits** in
steps of 32. Real arithmetic uses round-to-nearest, ties-to-even.

**Author: Nikolay Gromov.**

Previously named LimbForge. Existing `limbforge` headers, namespace, CMake package,
targets and configuration names remain supported for production consumers. The
GitHub repository is now [vanbaalon/WolfNum](https://github.com/vanbaalon/WolfNum).
See the [naming and compatibility notes](docs/naming.md).

> **Versioned API 1.x:** the compatibility baseline is **1.0.0**; the current release is
> **1.3.1**, releasing the WolfNum name with the existing production API.
> Production consumers can pin `v1.3.1` and rebuild with matching headers/archive;
> see [versioning and compatibility](docs/versioning.md).
> This README describes **current `main`**, including unreleased 2048-bit support,
> device/shape preflight and GPU-required GEMM. The `v1.3.1` tag supports 64–1024 bits
> and does not include those additions. Main still carries the 1.3.1 version macros;
> pin a commit when using unreleased features. See the [changelog](CHANGELOG.md).
> CPU and physical GPU validation use independent MPFR operations, with device validation
> currently covering an Apple M5 Max. See the [numerical contract](docs/numerics.md).

## GPU versus CPU

**82.3× faster than serial MPFR and 6.7× faster than 18-worker MPFR** in the
recorded 1024-bit multiplication-chain benchmark on an Apple M5 Max.
The GPU processes 65,536 independent values through 16 rounded multiplications
each in **3.94 ms**, including input/output transfers, encoding, submission,
and waiting.

The same machine, precision, and batch size also show gains for individual
operations:

| 1024-bit workload | Serial MPFR (ms) | 18-worker MPFR (ms) | GPU wall (ms) | vs serial CPU | vs 18-worker CPU |
|---|---:|---:|---:|---:|---:|
| Real multiplication | 18.39 | 1.90 | 1.62 | **11.3×** | **1.17×** |
| Real division | 40.83 | 3.60 | 2.09 | **19.6×** | **1.73×** |
| Complex division | 231.63 | 18.84 | 7.96 | **29.1×** | **2.37×** |
| 16-step multiplication chain (resident) | 324.10 | 26.38 | 3.94 | **82.3×** | **6.70×** |

Measured October 7, 2026; medians of nine warmed samples. GPU wall times include
transfers; compilation, decimal conversion, and buffer allocation are excluded.
CPU MPFR variables are preallocated, and results are checked against MPFR with
the same real rounding steps. Large batches and data reuse offer the strongest
gains; CPUs can be faster for small batches and simpler operations. See
[full methodology and measurements](docs/performance.md) and the
[raw benchmark data](benchmarks/results/final.csv). Those headline numbers preserve
the Round 1–5 snapshot; [later rounds](docs/optimizations.md) include workgroup
tuning, square, square root, and cooperative reductions, with their own raw
measurements.

## Features

- Correctly rounded real addition, subtraction, multiplication, division, square, and square root.
- Complex addition, multiplication, and division composed from rounded real primitives.
- Fused real `fma`/`fms` and complex `complex_fma`/`complex_fms`, each component rounded once.
- Exact widening and round-to-nearest narrowing casts between any two supported precisions.
- Broadcast (stride/period) operands for element-wise operations, so shared coefficients need no copies.
- Batched 4×4 complex LU with deterministic pivoting: solve, inverse and determinant per matrix.
- Segmented real/complex dot products with a fixed, reproducible fused order and optional shared table.
- Strided real/complex batched GEMM, device-generated power moments, matrix triple products,
  polynomial-source recurrences, and resident damping trials/solves (`batched_linalg.hpp`).
  See [contracts, usage and audit coverage](docs/section9.md) and the measured workloads below.
- Optional `PowerStorage::compact` power panels (since 1.1.0), reusable polynomial-source
  tables and explicit `PolynomialEvaluation::shared_sources` recurrences (since 1.3.0).
  Existing calls retain full-table storage and per-lane source evaluation.
- Dense real `syrk` (AᵀA) and `gemm` with one rounding of each exact dot product (`linalg.hpp`): exact
  integer GEMMs over exponent bands on TensorOps int8 residues, with an exact host fallback; updates
  `C - op(A) B` rounded once, and a blocked Cholesky factorization with multiple-right-hand-side
  triangular solves built on them (a fixed, reproducible rounding sequence).
- **Unreleased:** synchronous `Linalg::gemm_gpu` requires Metal execution for every nontrivial
  exact dot product, rejecting host fallback before submitting commands or modifying C.
  Zero/status outputs may still be assembled on the host; ordinary `gemm` retains its fallback.
- **Unreleased:** `Linalg::gemm` with a `GemmSpec`: submatrix views (offsets and leading dimensions) on host
  arrays and Buffers, `transpose_b`, overwrite/add/subtract and lower- or upper-only output. Adds a
  one-rounding symmetric/skew `syr2k`, large complex `gemm_complex`/`syr2k_complex` (with conjugation) and multithreaded host `negate_block`/`copy_block`/`axpy_block`; see
  [strided, signed and triangular products](docs/numerics.md#strided-signed-and-triangular-products).
- **Unreleased:** blocked LU with partial pivoting (`Linalg::factor_lu`): solves and determinants with a
  documented rounding sequence and GPU trailing updates; see [LU factorization](docs/numerics.md#lu-factorization).
- **Unreleased:** Pfaffians of real skew-symmetric matrices (`Linalg::pfaffian`, blocked Parlett–Reid with GPU skew
  updates); see [Pfaffian](docs/numerics.md#pfaffian).
- Real and complex Householder QR, optional column pivoting, reusable least-squares factors,
  multiple-right-hand-side solves and Q application (`linalg.hpp`). QR uses host panels
  and GPU/host updates; see [QR contracts](docs/numerics.md#qr-factorization-and-least-squares).
- **Unreleased:** allocation-free width/shape preflight in `capabilities.hpp`, device limits,
  matrix-memory estimates and placement diagnostics. Structural support does not guarantee
  available memory or GPU placement; see [wide-precision integration](docs/wide-precision.md).
- Batched real/complex polynomial values and Taylor jets `(p, p', p''/2)` by simultaneous Horner, with
  coefficient sets shared by adjacent points and composed or fused steps (`numerics.hpp`).
- Segmented norms (`norm_inf`, componentwise `norm_max`, exponent-scaled `norm2` on a fixed tree),
  `scaled_residual` and `summarize_status`: exact tie rule (lowest index), failing entries counted and
  never dropped from a norm.
- Correctly rounded element-wise `exp`, `expm1`, `log`, `log1p`, `sin`, `cos`, `atan2` and complex `exp`/`log`
  (`transcendental.hpp`, bit-identical to MPFR/MPC): certified GPU evaluation with GPU retry rungs at more limbs (host only for the rare cases beyond 35 words),
  plus complex integer powers by a documented fused sequence.
- Typed resident buffers, asynchronous submissions, and dependent operations in one command batch, including
  polynomials, norms, transcendental functions and dense `syrk`/`gemm` (`docs/execution.md`, "Resident units").
- In-place pointwise arithmetic, cached pipelines, and explicit buffer ownership checks.
- Algebra prewarming (synchronous/asynchronous), reusable product/normal-equation workspaces,
  workspace accounting/release, and explicit exact/sequential contract aliases (since 1.2.0).
- Fixed-order real and complex tree reductions with GPU-resident intermediate levels.
- Configurable threadgroup sizes and per-pipeline limits for measured tuning.
- A generic four-state complex recurrence primitive (with cooperative multi-lane execution for small batches), and a batched four-component vector recurrence (rank-one, matrix, affine, tangent and all-steps forms).
- Optional MPFR/MPC conversion through decimal strings or direct limb copies, with multithreaded `mpfr_t[]` / `mpc_t[]` array versions.
- Runtime-width bridges, GPU import from external inline MPC significands, asynchronous multi-submission
  waiting, and an offline measured CPU/GPU dispatch table.
- Reproducible benchmarks against both serial and multicore MPFR.

The GPU library depends on Metal and Foundation. MPFR/GMP are needed for the
optional conversion header, tests, benchmarks, and examples.

## Build and test

Use macOS 15 or newer, a Metal GPU, and Apple Command Line Tools:

```sh
git clone https://github.com/vanbaalon/WolfNum.git
cd WolfNum
brew install cmake mpfr gmp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
./build/limbforge_division
./build/limbforge_resident
./build/limbforge_reduction
```

Shaders are embedded and compiled at runtime. The command-line Metal compiler
and a full Xcode installation are unnecessary. Link `LimbForge::limbforge` in a
CMake consumer; [building and installation](docs/building.md) covers integration,
CPU-only validation, and building the GPU library without MPFR.

## Keep a computation on the GPU

```cpp
#include <limbforge/engine.hpp>
#include <limbforge/mpfr_bridge.hpp>
#include <vector>

using namespace limbforge;
using F = Float<384>;
Engine gpu;
std::vector<F> values(65536, from_decimal<384>("1"));
std::vector<F> factors(values.size(), from_decimal<384>("1.01"));
auto x = gpu.make_buffer<F>(values.size());
auto y = gpu.make_buffer<F>(values.size());
x.upload(values.data(), values.size());
y.upload(factors.data(), factors.size());

auto batch = gpu.batch();
for (int step = 0; step < 16; ++step)
    batch.run(Operation::mul, x, y, x);
auto submission = batch.submit();
// Do independent CPU work while the GPU executes.
auto timing = submission.wait();
x.download(values.data(), values.size());
```

Use `Buffer<Complex<12>>` for 384-bit complex values. Buffers stay resident across
batches. Mapping and transfers are rejected until the submission is waited;
previously obtained pointers also require caller coordination. See
[execution, ownership, and timing](docs/execution.md) and the complete
[resident example](examples/resident.cpp). A [norm example](examples/reduction.cpp)
combines square, tree sum, and square root in one GPU submission.

Unary operations use `batch.run(Operation::sqrt, x, x)` or
`batch.run(Operation::square, x, x)`. To sum a resident array, allocate a
one-element output and call `batch.tree_sum(x, total)`; each adjacent pair rounds
separately. See the [rounding contract](docs/numerics.md).

For a single operation, `Engine::run(bits, op, a, b, out, count)` provides a
synchronous host-array interface; `run_unary(bits, op, a, out, count)` handles
square and square root. Supply precise inputs through decimal strings
or MPFR; converting an existing `double` cannot restore lost digits.

## Precision and validation

| Significand bits | Approximate decimal capacity |
|---:|---:|
| 128 | 39 digits |
| 256 | 77 digits |
| 384 | 116 digits |
| 1024 | 308 digits |
| 1152 | 346 digits |
| 1536 | 462 digits |
| 2048 | 616 digits |

Wider arithmetic, GEMM/QR, device preflight and memory planning are described in
[wide-precision integration](docs/wide-precision.md). Transcendental APIs retain
their separately validated 1024-bit limit.

Capacity does not guarantee final numerical accuracy: cancellation and
conditioning may require guard precision. The separate binary exponent ranges
from −1,000,000,000 through +1,000,000,000. Arithmetic errors produce explicit
status bits. There are no NaNs, infinities, signed zeros, or subnormal encodings.

Complex arithmetic preserves rounding after each real primitive; it does not
promise a single correctly rounded MPC complex result.

The original arithmetic suite covers **761,856 real-operation cases across the 31
widths from 64 through 1024 bits**,
complex operations and recurrences, dense 65,536-value complex batches with
repeated GPU execution, and resident-buffer ownership and lifetime checks.
The original reduction tests cover those widths, odd tails, cancellation, statuses,
barriers, and scratch-buffer lifetimes. One million scalar quotient/remainder
cases validate the reciprocal used by
division. MPFR provides independent arithmetic references.

Separate wide-precision sweeps cover all **32 additional widths from 1056 through
2048 bits** for arithmetic, exact and batched complex GEMM, real/complex QR, casts
and tiny norms. Key widths 1056, 1152, 1536 and 2048 also pass Metal shader validation.
The [wide-precision guide](docs/wide-precision.md#verification) records test commands
and coverage; transcendental validation remains limited to 64–1024 bits.

## Benchmark and optimization rounds

```sh
./build/benchmark_limbforge > results.csv
./build/benchmark_limbforge --bits 384 --count 65536 --operation div --repeats 9
python3 benchmarks/run_round.py my-round
python3 benchmarks/compare.py benchmarks/results/baseline.csv results.csv
```

The default matrix covers three precisions, three batch sizes, real and complex
operations, and 16-step multiplication chains. It reports device and wall times,
serial MPFR, and a persistent multicore MPFR worker pool. Every result is checked
against MPFR. The round runner builds, tests, and benchmarks in order, preserving
logs and stopping on failure.

The original baseline and every optimization round are committed under
[benchmarks/results](benchmarks/results). Resident batches substantially reduce
repeated transfers and submission costs; exact reciprocal division and reduced
addition storage improve arithmetic throughput. Improvements depend on precision,
batch size, and workload. [Performance measurements](docs/performance.md) include
raw records and limitations; the [optimization log](docs/optimizations.md) also
records rejected experiments.

Recent measurements and integration checks are recorded separately from the headline benchmark:

- The [GEMM measurement grid](benchmarks/experiments/section9_gemm_extended/README.md)
  covers 352/384/448-bit Fourier and 10,000/100,000-matrix 4×4 batches, host/resident
  layouts and 120 offline calibration keys. These are profile-specific measurements;
  idle-consumer timing and automatic dispatch defaults remain open.
- The [J3 consumer sweep](benchmarks/experiments/section9_converged_sweep.md) passes
  CPU/GPU convergence checks at g = 0.1, 0.2 and 0.5 with opt-in GPU paths and the
  original tolerance. These fixtures establish correctness, not solver speedups.
- The [exact batched 4×4 prototype](benchmarks/experiments/exact_gemm4/BASELINE.md)
  passes its numerical gates but takes 1.054–4.582× composed GEMM wall time across
  24 profiles. It remains uninstalled and does not change public APIs or defaults.
- The [1152-bit reconstruction replay](docs/wide-precision.md#require-gpu-execution-for-reconstruction-gemms)
  verifies GPU-required GEMM on production-sized g=100 operands with zero host
  products and 80 sampled MPFR component checks. Its loaded-host timings cover
  reconstruction only, not the full solver.

## Scope

The current backend is Apple Metal, with fixed precision per dispatch. Fused
multiply-add, cooperative recurrence paths and certified transcendental functions
are implemented; their contracts and validation limits are documented above.
Further work includes validation on more devices, idle-consumer performance and
memory measurements, and optimizations that preserve each API's rounding sequence.
Grouped exact updates, native Gauss and exact batched 4×4 candidates remain
experiments, with no promotion to production APIs or defaults. Alternate layouts
have also been measured experimentally; the public representation remains an array of structs.
GPU multiprecision has prior art; [related work](docs/prior-art.md) explains the
context and the Metal target.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the validation and measurement workflow.
Authored by **Nikolay Gromov** and maintained by
[vanbaalon](https://github.com/vanbaalon); see [AUTHORS.md](AUTHORS.md).
WolfNum is distributed under the [MIT license](LICENSE).
