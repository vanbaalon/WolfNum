# Changelog

## Unreleased

- Speed up dense exact products (gemm, syrk, factorization updates; host and resident): reconstruction
  uses the CRT weighted sum with a fixed-point quotient instead of Garner's quadratic mixed-radix
  steps. It gives the same exact integer, so results are unchanged. Measured on a loaded M5 Max
  (best of 3, K = 32): 1.6× (128 bits), 1.8× (192), 2× (512) and 1.17× (1024) lower product GPU
  time. Add `LIMBFORGE_LINALG_STAGES` (profiling only) and `LinalgReport::digits_seconds`,
  `residue_seconds`, `reconstruct_seconds`. Validation: dense product, GEMM spec, resident,
  Cholesky, QR, complex QR, LU and Pfaffian suites pass bit for bit against MPFR. Products written
  in place with every line on the GPU (the factorization updates) skip the per-entry host assembly:
  the update's host overhead in a Pfaffian (n = 4096, block 64) fell from 0.7–0.8 s to 0.1–0.3 s.
  The LU and Pfaffian panels run one parallel pass per column (step) instead of two, with the same
  operations: Pfaffian panel 0.89 → 0.63 s (n = 4096, block 64), real LU panel 17–19% shorter.

- Add two-level blocking for Pfaffians (`EliminationOptions::inner_block`, opt-in, rounded up to even;
  previously rejected). Inner blocks are flushed into the panel as exact triangular products. An
  exchange with an index beyond the panel recomputes the entries that cross between panel and
  trailing matrix from block-start copies, so values depend only on `block` and `inner_block`, and GPU
  and host agree. Measured (192 bits, random entries): block 256 with inner 64 halves the wall time
  against block 32 at n = 4096 and 8192, and the panel is now the larger part. Validation: the MPFR
  replay covers inner blocks 1–6 with bounds, `stop_at`, separate calls, odd n, zero pivots and
  status entries, and a two-level Pf is checked against the one-level value.

- Add Buffer forms of `factor_lu`, `factor_lu_complex` and `pfaffian` (real or complex buffers),
  bit-identical to the host-array calls (tested). Record a width table (128–256 bits) for products,
  Pfaffians and complex LU in `docs/performance.md`.

- Add mixed-precision iterative refinement `LUFactor::solve_refined` (`LIMBFORGE_REFINEMENT_API`): a
  lower-width factor, exact full-width residuals (gemm / gemm_complex) and widened corrections, with
  per-step residual exponents. Validation: refinement from 128/256-bit factors to 256/1152 bits,
  real and complex, agrees with full-width solves to 2^-(bits-40). Measured: not faster than a direct
  192-bit solve with n/2 right-hand sides (see docs).

- Add broadcast reads of operand `a` (`LIMBFORGE_BROADCAST_A_API`: `Engine::run`/`run_ternary` and
  `CommandBatch::run` overloads with `a_index`), so kernel matrices f(x_i, y_j) build on the GPU from
  row and column vectors. Add the batched block-diagonal congruence `N = T M T^T`
  (`Engine::congruence`, `CommandBatch::congruence`, `LIMBFORGE_CONGRUENCE_API`): b x b complex
  blocks, two cfma chains per output, complex or real-part output, optional (strict) triangles.
  Validation: `tests/test_broadcast.cpp` (extended) and `tests/test_congruence.cpp` (new) against
  MPFR, host and resident; existing element-wise kernels are unchanged (fused and resident-unit
  suites pass).

- Add opt-in two-level blocking for LU (`EliminationOptions::inner_block`). Lazy panel dots span an
  inner block, and its terms are flushed as exact products; the trailing GPU update keeps the full
  block (complex LU, 192 bits, n = 2048: 3.65 s to 3.12 s). Add the
  `LIMBFORGE_ELIMINATION_API` feature macro and a double filter for the exact complex pivot
  comparison (decisions unchanged). Validation: the LU replay covers inner blocks of 1, 3 and 4
  with bounds, `stop_at` and separate-call equality.

- Add complex LU and complex Pfaffians (`factor_lu_complex`, `pfaffian_complex`). Complex dots round each
  component once, pivots compare |z|^2 exactly, and the scaled complex division is documented. The
  complex solve uses GPU block updates for many right-hand sides, and the determinant keeps its phase.
  Add pivoting segments and partial elimination (`EliminationOptions::bounds`, `stop_at`): leading
  determinants and Pfaffians at caller bounds from one factorization, and Schur complements
  `D - C A^-1 B` / `R + Q^T P^-1 Q`. `factor_lu` and `pfaffian` now take `EliminationOptions`
  (both unreleased). The LU and Pfaffian drivers share one real/complex implementation; real results
  are unchanged.
  Validation: the LU and Pfaffian MPFR replays now cover complex, bounds and `stop_at` at 256/1152 bits,
  bit for bit; leading values equal separate calls; the Schur product identities, |det|^2 against
  the real embedding and LU against QR solutions agree to 2^-(bits-40).

- Add Pfaffians of real skew-symmetric matrices (`LIMBFORGE_PFAFFIAN_API`): `Linalg::pfaffian`, blocked
  Parlett-Reid with pivoting and GPU skew trailing updates (triangular exact products). The rounding
  sequence depends only on the block size. Validation: `tests/test_pfaffian.cpp` (new) replays it in
  MPFR at 256/1152 bits bit for bit, with GPU/host agreement and Pf^2 = det(A) to working accuracy.

- Add blocked LU with partial pivoting (`LIMBFORGE_LU_API`): `Linalg::factor_lu`, `LUFactor` with
  `solve`, `determinant`, permutation and sign. The documented rounding sequence (one exact-dot
  rounding per block, as Cholesky) depends only on the block size; trailing updates run on the GPU.
  Validation: `tests/test_lu.cpp` (new) replays the sequence in MPFR at 256/1152 bits, bit for bit,
  with GPU/host agreement; Cholesky and QR tests pass after the shared `trsm` gained a unit-diagonal
  mode.

- Add strided, signed and triangular products (`LIMBFORGE_GEMM_SPEC_API`): `Linalg::gemm` with a
  `GemmSpec` (leading dimensions and offsets on host arrays and resident Buffers, `transpose_b`,
  overwrite/add/subtract, lower/upper-only output). Add `Linalg::syr2k` for the symmetric and skew
  rank-2k updates with one rounding per entry, and multithreaded exact `negate_block`/`copy_block`
  and one-rounding `axpy_block`. Existing entry points, defaults and rounding are unchanged;
  internally, strided host views not in a resident buffer are now packed rather than rejected.
  Add `Linalg::gemm_complex` and `syr2k_complex` for large complex host arrays with the same views,
  updates and triangles plus conjugation (`GemmSpec::conjugate_a/b`). Each real and imaginary part
  is one rounding of its exact dot, computed as one packed real product (2.6x the real GPU time at
  256 bits, n = 1024, k = 256).
  Triangular products also skip int8 residue tiles outside the triangle (0.57-0.60x the full GPU
  time at 256 bits, k = 256, n = 1024/2048 on a loaded host).
  Validation: `tests/test_gemm_spec.cpp` (new) against MPFR at 256/1152 bits on GPU, forced-host
  and resident paths. Existing `linalg_gpu`, `gpu_required_gemm`, resident (`--quick`), Cholesky
  and QR tests pass.

- Refresh the README for the renamed GitHub repository, distinguish the v1.3.1
  release from unreleased 2048-bit/preflight/GPU-required GEMM features, and document
  existing QR, storage, source-sharing and workspace APIs. Correct stale scope and
  validation claims and link the recent calibration, consumer and prototype results.
  Update current naming and release-clone guidance. Documentation validation checks
  relative links, API names, recorded evidence and diff formatting; no numerical change.

- Add `Linalg::gemm_gpu`: exact synchronous GEMM that rejects data-dependent
  host products before modifying C or submitting commands. Existing GEMM defaults
  remain unchanged. Extend the strong-coupling adapter with measured exponent-band
  limits, bounded row tiles, placement counters and `SC_GPU_REQUIRE_GEMM`.
  Production-sized g100/P640/Q540 reconstruction replay at 1152 bits confirms
  15,153,240 and 12,789,240 GPU outputs with zero host products, with 80 sampled
  MPFR component checks. GEMM calls took 25.727/16.030 s under concurrent CPU
  load; full-solver timing is not inferred. Strict API checks at 512/928/1024/1152/1536/2048
  pass normally and under Metal validation, including rejection before writing C.
  Focused CPU/GPU GEMM, resident GEMM, real QR and version/package tests pass;
  complex QR checks 528 factorizations/solves and 1056 Q applications against MPFR.

- Extend storage, bridges, arithmetic, dense/batched GEMM and real/complex QR
  from 1024 to 2048 bits at every 32-bit step. Wide Metal arithmetic uses rolled
  loops; exact host products and private batched/cast scratch accommodate the
  larger limb counts. Exact-window additions preserve wide composed complex
  arithmetic and norm-tree roundings; wider reductions respect threadgroup
  memory limits. Existing 64–1024-bit contracts remain unchanged;
  transcendental functions retain their 1024-bit validation limit.
- Add allocation-free structural width/shape queries in `capabilities.hpp` and
  `Engine::device_limits`, including overflow, indexing, stride and device buffer
  checks. Document QR matrix memory, reusable-factor iterative refinement and
  fallback diagnostics in `docs/wide-precision.md`.
- Validate all 32 new widths against MPFR/MPC for arithmetic, exact GEMM, cached
  complex GEMM, real/complex QR, casts and tiny norms. Key widths 1056, 1152,
  1536 and 2048 also pass Metal shader validation; the strong-coupling adapter
  passes small end-to-end products and tiny-pivot solves at 1152/1536/2048 bits.
  The complete build, 8 CPU tests and 20 GPU/integration regressions pass.

- Retained an uninstalled exact batched real/complex 4x4 prototype and complete
  checked baseline: all 31 supported widths, small/dense/supplement/multi-call
  normal/Metal checks, 26 normal/19 Metal suites, 24 four-contract comparisons
  and three clock probes pass. Independent raw/provenance verification retains
  both collector corrections. The exact prototype loses to composed in every
  profile (1.054–4.582x wall), so no public API/default change is promoted.
  Production remains WolfNum 1.3.1 / API 1; overhead optimization is pending.

- Completed the GEMM measurement grid at 352/384/448 bits, host/resident and
  CPU-interleaved/verified-warm-call layouts for Fourier and 1e4/1e5 4x4 batches.
  Thirty-two new checked records plus eight originals yield 120 offline keys;
  raw-sample/coverage checks, compiled factory and provenance/idle/overwrite
  guards pass. Library/API/default remain 1.3.1; idle consumer policy remains open; the subsequent fair exact4 baseline is
  recorded above.

- Completed the requested J3 g=.1/.2/.5 combined-switch convergence sweep.
  All final CPU/GPU modes solves preserve gtol=1e-22, match two history entries,
  agree in Delta below1e-25 and exercise every selected GPU path without fallback.
  Independent raw-output and frozen-identity checks pass. Collocation is seed
  preparation only; its stalled .5 result is retained separately. Idle timing,
  memory and damping/base work remain open; no library/default/version change.

- Archived an independently verified converged J3 g=.2 CPU/GPU comparison for
  the matching 1.3.1 consumer with exact-normal/Fourier/product3/LU4 switches on.
  Original gtol=1e-22 is preserved; two history entries match, residuals are
  1.48e-25 and maximum component Delta error is 3.05e-89. Full sweep and idle
  timings remain separate requirements; no library/default/version change.

- Archived a grouped exact trailing-update prototype with all 31 normal/Metal
  direct/factor/full-band references and complete 26/19 regression gates. The
  first checked n944/352-bit profile improves factor/solve wall time about 1.07x
  over the exact loop and misses the 3x factor target. Source, patch, raw timings,
  memory scopes and independent verification are retained; no API promotion.

- Added a rotating `--paths` subset selector to the opt-in complex GEMM benchmark.
  Defaults and the single `--path` selector remain available. All four CPU
  references still run; only selected GPU layouts are allocated and timed.
  Eight invalid-selector checks, all 31 CPU widths and 16 focused normal/Metal
  GPU checks pass against the unchanged 1.3.1 library. No public API/version or
  numerical default change.

- Retained the native per-product Gauss experiment: all-width normal/Metal and
  broad regression checks pass, but all 22 checked Fourier/4x4 profiles lose to
  composed GEMM. Archived the complete source, rotating-subset benchmark,
  correctness provenance and verified raw measurements; no production API or
  default change is accepted.

- Archived isolated Fourier, tangent product3 and batched LU4 consumer prototypes
  with completed MPFR/Metal checks and independently verified saved J3 GPU
  replays. LU4 covers all supported widths plus 374-bit consumer precision.
  Integrated these adapters locally behind off-default switches in a separate
  matching 1.3.1 consumer build. Focused normal/Metal helpers and default/combined
  J3 replays pass; original CPU results are preserved exactly. The requested g
  sweep and idle timings remain pending; production library/API/defaults stay at 1.3.1.

- Retained rejected 8x8 complex register-blocking attempts and all-width tile
  evidence; no numerical kernel/default or performance acceptance.

- Archived and independently verified matching 1.3.1 qscmx replay records: J=3
  g=.202 converges at its input tolerance; the historical J=2 g=.2 diagnostic
  remains stalled. No new consumer switch, complete g sweep or timing acceptance.

- Recorded 14 offline GEMM calibration keys from checked WolfNum 1.3.1 Fourier
  and resident 4x4 profiles. Generated C++ lookups and failed/nonterminal/hash, idle
  and overwrite guards pass. Busy-host results establish no consumer default.

## 1.3.1 — 2026-10-09

- Renamed the project to **WolfNum**, the numerical library in the Wolfbook family.
  Updated documentation, comments, diagnostics and benchmark method labels. The
  CMake project is now `WolfNum`; former project variables remain available.
  Existing `limbforge` headers, C++ symbols, version macros, package, build targets,
  options and environment variables retain their production-compatible names.
  Repository URLs, published tags and recorded benchmark evidence retain their
  original identities. Numerical behavior and API major remain unchanged.
  Validation: rebuilt the affected library, example and benchmark targets; all
  three focused CPU version/package/bridge checks pass. An embedded consumer
  also confirms legacy targets and directory variables remain available. Core
  arithmetic and shader sources are byte-identical to the pre-rename baseline.
  Final 1.3.1 validation: all seven CPU checks and both linked GPU smoke checks
  pass, as do version/package checks after rebuilding the main checkout. Fourteen
  numerical implementation files are byte-identical to 1.3.0. Experimental all-width
  and 26/19 regression gates are retained separately; no prototype API is promoted.

- Added opt-in normal-equation comparisons and tightened the GEMM benchmark worker
  and timing scopes. Each timed CPU/GPU output is checked against its documented
  contract. Recorded six normal-equation profiles and scoped exact-path guidance;
  an offline exporter adds checked, profile-specific calibration without changing defaults.
  Exact-trial prototypes and their measurements remain experimental;
  they are not installed public APIs in this release.

## 1.3.0 — 2026-10-08

- Added `polynomial_sources` for reusable device p/q tables and explicit
  `PolynomialEvaluation::shared_sources` recurrence overloads. Staged rounded Horner
  states and dot/update passes preserve order, tails, statuses and ownership. Existing
  calls retain per-lane evaluation.
- Added separate `PolynomialPrewarm` requests and synchronous/asynchronous preparation
  methods; the existing `BatchedPrewarm` layout and calls stay compatible.
- All 31 source/recurrence widths pass normally and under shader validation, with
  26/26 normal and 19/19 GPU-validation suites passing. Final API, concurrency, lifetime,
  linked fixtures and version/package checks pass in their focused scopes.
- Correctness-checked 352/384/448-bit measurements show 22–63x lower wall time than
  the old GPU recurrence on the recorded shapes. CPU ratios remain provisional under
  busy-host load; no solver or universal CPU speedup is claimed. Added an offline
  exporter and 50 profile-specific break-even keys; other-operation calibration remains
  pending. See `benchmarks/experiments/section9_shared_sources.md`.

## 1.2.0 — 2026-10-08

- Added reusable intermediates for `BatchedLinalg::product3` and
  `normal_equations_exact`, with `workspaces()` and `release_workspaces()` queries.
  Busy storage survives release until its owning batch is destroyed or waited.
  The arithmetic sequence, provisional-output rules and existing signatures are unchanged.
- Added operation-aware `BatchedPrewarm` and shape-aware `LinalgPrewarm`
  requests, with synchronous and asynchronous preparation on the owning algebra cache.
  Background tasks retain pipeline/table state and the Metal device, while Linalg host
  scratch/worker lifetime stays on its owner thread. Cache access is synchronized;
  numerical shader sources, public layouts, defaults and rounding sequences are unchanged.
- Added explicit contract names: Linalg exact GEMM/SYRK and blocked
  Cholesky/solve; BatchedLinalg sequential GEMM, normal equations and damping
  factor/solve. Existing entry points remain supported with identical defaults,
  validation, status and completion behavior. No arithmetic/backend changes.
- Validation: all three additions pass their combined focused physical-GPU checks
  normally and under shader validation, including busy workspace release, distinct
  exact fallback repairs, prewarm/encode overlap, detached cache lifetime and an
  independent cancellation example distinguishing exact from sequential dots.
  Matching 1.2.0 header/archive and installed-package checks pass 2/2. No numerical
  shader or core arithmetic changed; these are focused host/API checks, not a new
  full arithmetic audit. Raw logs and build hashes are retained. No speedup claim.

## 1.1.0 — 2026-10-08

- Added `PowerStorage::compact` overloads of host-array and resident `power_moments`.
  An eight-row power panel and an unscaled checkpoint replace the full auxiliary table
  when requested. Existing entry points retain full-table storage; public shape/number
  layouts and all rounding/ownership contracts are unchanged. Compact scratch is 6.7–11.2×
  smaller at the measured large shapes, with additional launch costs. Most measured cases
  are slower, so this release makes no power-kernel speedup claim.
- Recorded interleaved full-table/compact benchmarks at 352/384/448 bits, host and resident,
  with seven repeats and independent MPFR checks of initial and every timed output.
  These are busy-host library measurements, not accepted end-to-end solver timings.
- Validation: the numerical candidate passed 26/26 library suites, 19/19 shader-validation
  GPU suites and all-width dense/reference sweeps. The exposed compact API passes its
  31-width power reference sweep normally and under shader validation; focused tests check
  host/resident selection, negative powers, empty reductions and invalid enum rejection.
  Version/package checks are rebuilt for 1.1.0. Raw records distinguish candidate gates
  from final API checks.

## 1.0.1 — 2026-10-08

- Corrected dense complex `BatchedLinalg::gemm` outputs that could have incorrect limbs
  or signs in the original Metal kernels. The implementation uses a private arithmetic
  namespace with padded local storage, uniform component threads and explicit packed
  loads/stores. Logical precision, public layouts, defaults and rounding sequences are
  unchanged. This also corrects the GEMM stages of `product3` and `power_moments`.
- Expanded the opt-in MPFR/MPC audit with repeated dense GEMM, wide exponent gaps,
  damping factors and multiple RHS, polynomial sharing tails and dense inline imports
  at all 31 widths. Retained baseline failures and rejected workaround patches. The
  benchmark now checks every measured output, rather than only its warm-up output.

- Validation: 26/26 CTest suites and 19/19 GPU suites under shader validation pass.
  All 31 widths pass the complete expanded dense audit and broad-exponent GEMM checks,
  normally and under shader validation. Additional valid/status normal-equation and
  negative-power sweeps pass at all widths. Solver-size integration and performance
  calibration remain separate follow-up work.

- Added the section 9 follow-up review and optimization priorities. Aligned planning documents
  with the 1.x compatibility policy, clarified recorded test coverage, and documented power-tile
  rounding and recurrence-sharing constraints. Documentation only; no numerical behavior changes.
- Added an opt-in section 9 MPFR reference runner selecting all 31 widths, expanded arithmetic,
  status and shape cases, and repeated dense GEMM stress cases. The default smoke test remains
  light. The runner preparation itself changed no public API or kernels; the subsequent
  expanded audit exposed and led to the complex GEMM correction above.
- Consolidated section 9 numerical contracts and completion/storage rules into the canonical
  numerics and execution documents, with links from the usage guide and follow-up records.

## 1.0.0 — 2026-10-08

First versioned public API baseline, incorporating the development work through section 9.

- Public compile-time version/API macros, constexpr version comparisons, and linked-library
  version queries. An exact startup check detects mismatched headers and archive.
- One canonical version header drives CMake and installed package metadata. Compatible-major
  and exact version requests are supported, with an independent installed-consumer check.
- Documented source/numerical compatibility and deprecation policy, immutable release tags,
  production dependency pinning and contribution rules. Existing entry points are preserved.
- The baseline includes 64–1024-bit arithmetic, fused recurrences through 1024 bits, resident
  buffers, exact dense products, Cholesky/QR, polynomial and transcendental operations,
  batched products/power moments, polynomial-source recurrences, damping trials and bridges.
- Validation for this release metadata: focused CPU version and installed-package checks.
  Section 9 retains its recorded light GPU/MPFR smoke coverage; its full numerical/solver
  audit and large-workload calibration are pending. No new performance claims are made.

## 0.1.0 — Initial development version

- Apple Metal backend for fixed-precision floating-point arithmetic, 64–1024 bits.
- Real addition, subtraction, multiplication, and division; complex arithmetic.
- Fused four-term complex recurrences with shared coefficients.
- Optional MPFR conversion, independent arithmetic tests, and benchmarks.
- Baxter Q-propagation comparison against BSolver4D.
- CMake integration, installed package support, examples, and documentation.
