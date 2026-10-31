#pragma once
#define LIMBFORGE_GPU_REQUIRED_GEMM_API 1
#define LIMBFORGE_GEMM_SPEC_API 1
#define LIMBFORGE_LU_API 1
#define LIMBFORGE_PFAFFIAN_API 1
// Dense real products with one rounding per output (plan D6 via L1c): C[i][j] = RN(sum_k L[i][k]*R[k][j]),
// or RN(C[i][j] - sum_k ...) as an update, computed exactly by an integer GEMM over residues (Metal TensorOps
// int8) and rounded once, ties to even. Contract and algorithm: docs/numerics.md, "Dense products". Host arrays, and
// resident Buffer operands in an Engine's CommandBatch ("Resident products"); element = Number<bits/32> (Float<bits>), matrices row-major. Statuses propagate: C[i][j] = zero(OR of the
// statuses of row i of L and column j of R, and of the old C[i][j] for an update) when that OR is nonzero. Empty K
// gives canonical zero (an update leaves C unchanged). Blocked Cholesky, triangular solves and Householder QR build on these updates.
#include "core.hpp"
#include "engine.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
namespace limbforge {
struct LinalgOptions {
    int band_bits=64;      // G: a band holds exponents [top-G, top] of one row/column; entries are exact (bits+G)-bit integers
    int max_bands=4;       // rows/columns needing more bands use the exact host fallback
    int max_spread=512;    // ... as do rows/columns whose nonzero exponents span more than this (accumulator bound)
    unsigned host_threads=0; // fallback/assembly threads; 0 = hardware concurrency
    bool force_fallback=false; // every output through the exact host path (testing)
    int resident_bands=2;  // resident calls: workspace for up to resident_bands*lines band rows per side; beyond, the whole call is exact on the host at wait
};
// Exact-product pipelines also serve the trailing updates of Cholesky and QR.
// Resident calls additionally use analysis/planning pipelines and per-K modulus tables.
struct LinalgPrewarm {
    std::vector<int> bits;
    std::vector<std::size_t> inner_sizes; // resident K values, 0..65472
    bool resident=false;
};
struct LinalgPair { unsigned left_band,right_band,rows,cols,moduli; };
// Diagnostics of the last call. "Lines" are rows of the left operand and columns of the right operand.
struct LinalgReport {
    std::size_t m=0,n=0,k=0;
    std::vector<std::size_t> left_bands,right_bands; // [b] = lines with exactly b bands (index 0: zero or status lines)
    std::size_t left_fallback=0,right_fallback=0;    // lines routed to the exact host path
    std::size_t gpu_outputs=0,multi_band_outputs=0,fallback_outputs=0,trivial_outputs=0;
    std::vector<LinalgPair> pairs;                   // band pairs (sub-blocks of the extended product); SYRK lists b >= c
    std::size_t gemm_rows=0,gemm_cols=0,int8_gemms=0; // extended operand sizes (all bands); 3 int8 GEMMs per modulus
    bool zero_copy_output=false;                    // C was page aligned and written by the GPU in place
    bool resident_overflow=false;                   // resident: band rows exceeded the workspace; every output with no zero line on the host
    double analysis_seconds=0,upload_seconds=0,gpu_seconds=0,assembly_seconds=0,fallback_seconds=0;
};
// Memory a Linalg retains between calls (Linalg::workspaces / release_workspaces; docs/numerics.md, "Workspace release"), in bytes.
struct LinalgWorkspaces {
    std::size_t scratch_bytes=0;   // host-thread scratch: product staging, QR/Cholesky work matrices, W/Y blocks, right-hand sides
    std::size_t idle_bytes=0;      // resident-product workspaces (per call and per batch) held by no batch
    std::size_t busy_bytes=0;      // resident-product workspaces held by an unsubmitted batch or an unwaited submission (also released ones)
    std::size_t busy_workspaces=0;
};
namespace detail {
struct LinalgPass;
template<class T> constexpr int real_bits(){static_assert(Format<T>::bits&&!Format<T>::complex,"dense linear algebra uses real Float<bits> buffers");return Format<T>::bits;}
}
// Handle of a resident product (Linalg::syrk / gemm on a CommandBatch). C is final only after Submission::wait(), which also
// computes the outputs of host-fallback lines (docs/numerics.md, "Resident products"). Copies share state.
class LinalgTicket {
    std::shared_ptr<detail::LinalgPass> pass_;
    explicit LinalgTicket(std::shared_ptr<detail::LinalgPass> p):pass_(std::move(p)){}
    friend class Linalg;
public:
    LinalgTicket()=default;
    bool resolved() const;               // the submission was waited and the host-fallback outputs were written (C final)
    const LinalgReport& report() const;  // analysis and placement as the host call reports it (no timings); throws before resolved()
    bool provisional_reads() const;      // a later operation of the batch read C before wait; throws before resolved()
};
// Blocked Cholesky and triangular solves (docs/numerics.md, "Cholesky factorization"). The result depends on
// `block` (the documented rounding sequence) and on nothing else: which updates run on the GPU or the host
// changes only the speed. An update with fewer multiply-adds (outputs x K) than host_macs (factorization) or
// solve_host_macs (triangular solves) runs on the host.
struct FactorOptions {
    std::size_t block=32;          // column block nb; 0 selects a single block (every dot product exact over all k)
    double host_macs=5e4;          // factorization updates below this many multiply-adds use host exact dots (same results)
    double solve_host_macs=4e5;    // the same for triangular-solve updates (measured: the host wins for a few right-hand sides)
    bool gpu=true;                 // false: everything on the host (same results)
};
// Per-call outcome of cholesky. pivot = n on success; otherwise the first column p whose pivot
// s_pp = RN(a_pp - sum_k l_pk^2) is zero, negative or carries a status (then pivot_sign / pivot_status describe it).
// On failure L[i][j] is final for j < p, and every lower entry with j >= p is zero with status `invalid`.
struct CholeskyInfo {
    std::size_t pivot=0; int pivot_sign=0; word pivot_status=0;
    Timing timing{0,0};                       // gpu_seconds: summed command buffers; wall: whole call
    std::size_t blocks=0,gpu_updates=0,host_updates=0;
    double panel_seconds=0,update_seconds=0;  // host diagonal-block + panel work; trailing updates (GPU or host), wall
};
// Householder QR (docs/numerics.md, "QR factorization and least squares"): FactorOptions (block, GPU/host split; the
// result depends on block only) and an optional rank test: rank_bits > 0 also stops at the first column whose |r_jj|
// lies more than rank_bits binades below the largest earlier |r_kk| (exponent comparison, no rounding). With pivot, column
// pivoting (as LAPACK xGEQP3): step j takes the remaining column with the largest downdated squared norm (exact comparisons,
// ties to the lowest original column index), so A P = Q R with |r_jj| nonincreasing up to rounding; rank_bits then is the
// rank tolerance. Pivoted factors give basic (not minimum-norm) least-squares solutions when rank deficient.
struct QROptions : FactorOptions { int rank_bits=0; bool pivot=false; };
// Per-factor outcome. rank = n for a full-rank factor; otherwise the first column p (in order, after pivoting) whose norm
// is zero (zero_column), carries a status (status_column; status = its OR) or fails rank_bits (small_column). Reflectors
// 0 .. p-1 are then final, and the factorization stops there (see QRFactor::r).
struct QRInfo {
    enum Reason {full_rank=0,zero_column=1,small_column=2,status_column=3};
    std::size_t rank=0; int reason=full_rank; word status=0;
    Timing timing{0,0};                       // gpu_seconds: summed command buffers; wall: whole call
    std::size_t blocks=0,gpu_updates=0,host_updates=0; // trailing block updates (up to 3 products each) by placement
    double panel_seconds=0,update_seconds=0;  // host panels; trailing updates (GPU or host), wall
    std::size_t norm_recomputations=0;        // pivoting: downdated norms recomputed from the reduced column
};
// Strided, signed and triangular products (Linalg::gemm with a GemmSpec, Linalg::syr2k; docs/numerics.md, "Strided, signed and
// triangular products"). Every output is one rounding of its exact dot product, as gemm: overwrite C = RN(op(A) op(B)),
// add C = RN(C + op(A) op(B)), subtract C = RN(C - op(A) op(B)). The default GemmSpec is gemm(bits,false,A,B,m,n,k,C).
enum class GemmUpdate {overwrite,add,subtract};
enum class Triangle {full,lower,upper};
struct GemmSpec {
    bool transpose_a=false,transpose_b=false;     // op(A) = A (stored m x k) or A^T (stored k x m); op(B) = B (k x n) or B^T (stored n x k)
    std::size_t lda=0,ldb=0,ldc=0;                // row strides in elements of the stored row-major A, B and C; 0 = their column counts
    std::size_t a_offset=0,b_offset=0,c_offset=0; // first element of A, B and C, in elements from the pointer (or buffer start)
    GemmUpdate update=GemmUpdate::overwrite;
    Triangle triangle=Triangle::full;             // m == n: only outputs i >= j (lower) or i <= j (upper) are computed and written (about half the GPU work)
    bool require_gpu=false;                       // host arrays: as gemm_gpu (host fallback throws before C changes); Buffer forms reject it
    bool conjugate_a=false,conjugate_b=false;     // complex products only: conjugate op(A) / op(B) (with a transpose: A^H, B^H)
};
namespace detail { struct QRData; }
namespace detail { struct LinalgHooks; } class ComplexQRFactor; // complex QR (end of this header, src/linalg_complex.mm)
class Linalg;
// Blocked LU with partial pivoting (docs/numerics.md, "LU factorization"): P A = L U with unit lower L. FactorOptions as Cholesky
// (block, GPU/host split; the result depends on block only). Per-factor outcome: zero_pivot = the first column whose pivot is
// zero (n if none; the factorization continues as LAPACK xGETRF, U is singular); status_column = the first column whose pivot
// candidates carry a status (n if none; the factorization stops there: rows and columns >= status_column of the factor are zero
// with status invalid, and status is the OR of that column's candidate statuses).
struct LUInfo {
    std::size_t zero_pivot=0,status_column=0; word status=0;
    Timing timing{0,0};                       // gpu_seconds: summed command buffers; wall: whole call
    std::size_t blocks=0,gpu_updates=0,host_updates=0;
    double panel_seconds=0,update_seconds=0;  // host panels and U rows; trailing updates (GPU or host), wall
};
// Pivoting segments and partial elimination for factor_lu(_complex) and pfaffian(_complex) (docs/numerics.md, "Pivoting
// segments and partial elimination"). bounds: strictly increasing, in (0, n] (even for Pfaffians); the pivot for column
// (step) j is chosen only among rows (indices) before the first bound > j, so every leading submatrix [0, b) is factored on
// its own and its determinant / Pfaffian is reported. stop_at = k > 0 eliminates columns [0, k) only (pivots within [0, k),
// a block ends at k; bounds must not exceed k) and keeps the Schur complement in original order: D - C A^-1 B (LU) or
// R + Q^T P^-1 Q (skew, for A = [P Q; -Q^T R]). Without either option the factorizations are unchanged.
struct EliminationOptions : FactorOptions {
    std::vector<std::size_t> bounds;
    std::size_t stop_at=0;
};
// Pfaffian of a real skew-symmetric matrix by blocked Parlett-Reid elimination with pivoting (docs/numerics.md, "Pfaffian").
// FactorOptions: block = columns per block (rounded up to even; 0 = one block), host_macs and gpu as Cholesky; the value depends
// on block only. zero_pivot: the step column k whose pivot candidates are all zero (n if none; Pf = 0 exactly). status_column:
// the step column whose candidates carry a status (n if none; Pf = zero with status, the OR of those statuses). sign: det(P).
struct PfaffianInfo {
    std::size_t zero_pivot=0,status_column=0; word status=0; int sign=1;
    Timing timing{0,0};
    std::size_t blocks=0,gpu_updates=0,host_updates=0;
    double panel_seconds=0,update_seconds=0;
};
namespace detail { struct LUData; }
// Owning LU factor of an n x n matrix. Like QRFactor it copies what it needs, uses the Linalg that made it (which must outlive
// it) and follows the Linalg threading rule.
class LUFactor {
public:
    LUFactor(); ~LUFactor(); LUFactor(LUFactor&&) noexcept; LUFactor& operator=(LUFactor&&) noexcept;
    bool empty() const; int bits() const; std::size_t size() const; std::size_t block() const;
    const LUInfo& info() const; bool nonsingular() const;   // no zero pivot and no status column
    bool is_complex() const;                 // from factor_lu_complex: lu(), solve and determinant use Complex<bits/32>
    std::size_t eliminated() const;          // columns eliminated: size(), or EliminationOptions::stop_at
    const std::vector<std::size_t>& bounds() const;
    // det of each leading submatrix [0, b) for b in bounds(): bounds().size() values (Float<bits()> or Complex) written
    // to out, each the running pivot product of determinant() at that column with the permutation sign so far. After a zero
    // pivot in column j, bounds beyond j's segment get zero with status invalid (restricted pivoting cannot factor them).
    void leading_determinants(void* out) const;
    // The Schur complement of a partial elimination: (size() - eliminated())^2 entries, row-major, original row/column order.
    void schur(void* out) const;
    // Row-major Float<bits> (complex factor: Complex<bits/32>) n x n owned by the factor: strictly lower part = L (unit
    // diagonal not stored), upper part = U.
    const void* lu() const;
    const std::size_t* permutation() const;  // row i of P A is row permutation()[i] of A
    int permutation_sign() const;            // det(P) = +1 or -1
    // X (n x nrhs) solves A X = B: X = P B, then L Y = X (unit lower) and U X = Y, each blocked as trsm (docs/numerics.md).
    // A singular factor divides by its zero pivot (division_by_zero statuses); a status column gives zero with status invalid
    // everywhere. X must not overlap B.
    Timing solve(const void* B,std::size_t nrhs,void* X) const;
    // det(A) = permutation_sign() * u_00 * u_11 * ... * u_{n-1,n-1}, the product rounded after each multiplication in index
    // order (n - 1 roundings; empty matrix: 1; complex: each component of each product rounded once). Writes one Float<bits()>
    // (or Complex) to out. A status column gives zero with status invalid. After a partial elimination: det of the leading
    // eliminated() block. solve requires a complete factorization (logic_error otherwise).
    void determinant(void* out) const;
private:
    friend class Linalg; std::unique_ptr<detail::LUData> d;
};
// Owning QR factor of an m x n matrix (m >= n): Q = H_0 ... H_{n-1}, H_j = I - tau_j v_j v_j^T, and R. The factor copies
// everything it needs and never refers to A again: when A changes, factor it again (no caching by address). It uses the
// Linalg that made it (scratch buffers, GPU): that Linalg must outlive it, and calls follow the Linalg threading rule.
class QRFactor {
public:
    QRFactor(); ~QRFactor(); QRFactor(QRFactor&&) noexcept; QRFactor& operator=(QRFactor&&) noexcept;
    bool empty() const; int bits() const; std::size_t rows() const; std::size_t cols() const; std::size_t block() const;
    const QRInfo& info() const; bool full_rank() const;
    // Row-major Float<bits> arrays owned by the factor (valid while it lives): R (n x n, upper; with rank p < n, rows
    // i >= p hold the unreduced remainder in columns j >= p), V (m x n: column j is v_j, zero above row j, v_j[j] =
    // the scaled v0), tau (n) and T (blocks of nb x nb, the compact-WY upper triangles; block b at offset b*nb*nb).
    // With pivoting all of them belong to A P: column j of R and V is column permutation()[j] of A.
    const void* r() const; const void* v() const; const void* tau() const; const void* t() const;
    bool pivoted() const; const std::size_t* permutation() const;  // identity without pivoting
    // X (n x nrhs) = least-squares solution of min ||A x - b|| for each column b of B (m x nrhs): Q^T B, then R X = (Q^T B)[0:n)
    // (pivoted: x[perm[i]] = z_i). Rank deficient (rank p < n): an unpivoted factor writes zero with status invalid to every
    // entry of X; a pivoted one gives the basic solution x[perm[i]] = z_i from R[0:p,0:p] z = (Q^T b)[0:p) and x[perm[i]] = 0
    // for i >= p (LAPACK xGEQP3 + a truncated triangular solve; it is not the minimum-norm solution). A status column gives
    // zero with that status (and invalid) everywhere. X must not overlap B.
    Timing solve(const void* B,std::size_t nrhs,void* X) const;
    // B (m x nrhs) <- Q^T B (transpose) or Q B, in place.
    Timing apply_q(void* B,std::size_t nrhs,bool transpose) const;
    // The same on Buffer operands (Float<bits()>) of the Linalg's engine, synchronously outside any batch (Linalg buffer forms).
    template<class T> Timing solve(const Buffer<T>& B,std::size_t nrhs,Buffer<T>& X) const{return solve_buffers(detail::real_bits<T>(),detail::Access::operand(B),nrhs,detail::Access::operand(X));}
    template<class T> Timing apply_q(Buffer<T>& B,std::size_t nrhs,bool transpose) const{return apply_q_buffer(detail::real_bits<T>(),detail::Access::operand(B),nrhs,transpose);}
private:
    Timing solve_buffers(int bits,const detail::Operand& B,std::size_t nrhs,const detail::Operand& X) const;
    Timing apply_q_buffer(int bits,const detail::Operand& B,std::size_t nrhs,bool transpose) const;
    friend class Linalg; std::unique_ptr<detail::QRData> d;
};
// One Linalg per host thread (scratch buffers and compiled libraries are reused). C must not overlap A or B;
// page-aligned A, B and C are used by the GPU in place. Calls return after the GPU has finished.
class Linalg {
public:
    explicit Linalg(LinalgOptions options={}); ~Linalg();
    explicit Linalg(Engine& engine,LinalgOptions options={}); // compiles on the engine's Metal device (required for Buffer operands)
    Linalg(const Linalg&)=delete; Linalg& operator=(const Linalg&)=delete;
    void prewarm(const LinalgPrewarm&);
    std::future<void> prewarm_async(LinalgPrewarm);
    std::string device_name() const;
    // C (cols x cols) = A^T A for A (rows x cols). lower_only writes C[i][j] for i >= j and leaves the
    // rest of C untouched; otherwise the full symmetric matrix is written. With subtract, C[i][j] = RN(C[i][j] - (A^T A)[i][j])
    // with one rounding (lower triangle only: subtract requires lower_only).
    Timing syrk(int bits,const void* A,std::size_t rows,std::size_t cols,void* C,bool lower_only=true,bool subtract=false);
    // C (m x n) = op(A) B with B (k x n); op(A) = A (m x k) or, with transpose_a, A^T for A (k x m).
    // With subtract, C = RN(C - op(A) B), one rounding per entry.
    Timing gemm(int bits,bool transpose_a,const void* A,const void* B,std::size_t m,std::size_t n,std::size_t k,void* C,bool subtract=false);
    // Same exact rounding as gemm; every nontrivial dot must execute on Metal. Data-dependent host fallback throws
    // after analysis and before modifying C or issuing commands. Zero/status outputs may be assembled on the host.
    // Configure exponent bands with LinalgOptions; this entry point never loosens limits or changes precision.
    Timing gemm_gpu(int bits,bool transpose_a,const void* A,const void* B,std::size_t m,std::size_t n,std::size_t k,void* C,bool subtract=false);
    // C (m x n) per GemmSpec: views of larger row-major matrices, either operand transposed, overwrite/add/subtract and lower- or
    // upper-only output (the rest of C is untouched). C must not overlap A or B. Strided operands that are not inside a resident
    // buffer are packed once (multithreaded); a strided page-aligned C is used in place.
    Timing gemm(int bits,const void* A,const void* B,std::size_t m,std::size_t n,std::size_t k,void* C,const GemmSpec& spec);
    // Rank-2k update with one rounding per entry: S = A^T B + B^T A (symmetric) or A^T B - B^T A (skew) for A and B (k x n), or with
    // spec.transpose_a S = A B^T +/- B A^T for A and B (n x k); C (n x n) takes S per spec.update and spec.triangle (leading
    // dimensions and offsets apply; transpose_b must be false). Each entry is the exact 2k-term dot of [A; B]^T [B; +/-A], packed
    // once (O(nk) copies), so status lines follow gemm: entry (i, j) carries the statuses of lines i and j of A and B.
    Timing syr2k(int bits,const void* A,const void* B,std::size_t n,std::size_t k,void* C,bool skew,const GemmSpec& spec={});
    // Complex products on Complex<bits/32> host arrays (row-major; leading dimensions and offsets in complex elements): C (m x n)
    // per spec with op = transpose and/or conjugate. Each real and imaginary part is one rounding of its exact 2k-term real dot
    // (re = sum re*re - im*im, im = sum re*im + im*re), computed as one real product of the packed lines [re, im] (O((m+n)k)
    // copies, held in scratch); a status in row i of op(A) or column j of op(B) gives both parts zero with that status.
    Timing gemm_complex(int bits,const void* A,const void* B,std::size_t m,std::size_t n,std::size_t k,void* C,const GemmSpec& spec={});
    // Complex rank-2k update: S = A^T B + B^T A (symmetric) or A^T B - B^T A (skew) for A and B (k x n), or A B^T +/- B A^T with
    // transpose_a (A and B n x k); conjugate_a conjugates the left factors (A^H B +/- B^H A: Hermitian / skew-Hermitian). Each part
    // of each entry is one rounding of its exact 4k-term dot; C takes S per spec.update and spec.triangle (transpose_b,
    // conjugate_b must be false).
    Timing syr2k_complex(int bits,const void* A,const void* B,std::size_t n,std::size_t k,void* C,bool skew,const GemmSpec& spec={});
    // Exact elementwise host utilities on row-major Float<bits> blocks (rows x cols; leading dimensions in elements, 0 = packed),
    // multithreaded over the Linalg's workers. negate_block: X = -X in place. copy_block: Y = X, -X, X^T or -X^T (X is rows x cols,
    // or cols x rows with transpose); Y must not overlap X. axpy_block: Y = RN(Y + alpha X) per entry, one rounding (statuses OR).
    void negate_block(int bits,void* X,std::size_t rows,std::size_t cols,std::size_t ldx=0);
    void copy_block(int bits,const void* X,std::size_t ldx,void* Y,std::size_t ldy,std::size_t rows,std::size_t cols,bool transpose=false,bool negate=false);
    void axpy_block(int bits,const void* alpha,const void* X,std::size_t ldx,void* Y,std::size_t ldy,std::size_t rows,std::size_t cols);
    // Lower L (n x n, row-major) with A = L L^T from the lower triangle of symmetric positive definite A; the strict
    // upper triangle of L is set to zero. L may equal A (in place); otherwise they must not overlap.
    CholeskyInfo cholesky(int bits,const void* A,std::size_t n,void* L,const FactorOptions& options={});
    // X (n x nrhs) solves L X = B, or L^T X = B with transpose, for lower L (strict upper triangle not read).
    // X may equal B; otherwise X must not overlap B or L.
    Timing trsm(int bits,bool transpose,const void* L,std::size_t n,const void* B,std::size_t nrhs,void* X,const FactorOptions& options={});
    // X solves (L L^T) X = B: trsm, then trsm with transpose.
    Timing cholesky_solve(int bits,const void* L,std::size_t n,const void* B,std::size_t nrhs,void* X,const FactorOptions& options={});
    // LU with partial pivoting of A (n x n, row-major); A is only read. See LUFactor.
    LUFactor factor_lu(int bits,const void* A,std::size_t n,const EliminationOptions& options={});
    // Complex LU (A: Complex<bits/32>): the same sequence with complex exact dots (one rounding per component), pivots by exact
    // comparison of |s_ij|^2, and l_ij = s_ij / u_jj by the documented complex division (docs/numerics.md, "Complex LU and
    // Pfaffians"). The factor's solve runs on host threads.
    LUFactor factor_lu_complex(int bits,const void* A,std::size_t n,const EliminationOptions& options={});
    // Pf(A) for skew-symmetric A (n x n, row-major): only the strict lower triangle is read (A_ij = -A_ji is assumed). Writes one
    // Float<bits> to out: sign * a_0 * a_2 * ... (a_k = the k-th step's pivot entry), rounded after each multiplication. n = 0
    // gives 1; odd n gives zero with the OR of the lower entries' statuses. See PfaffianInfo and docs/numerics.md. With
    // EliminationOptions: out = Pf of the eliminated leading block [0, stop_at) (whole matrix by default); leading (if not
    // null) receives Pf of each leading submatrix [0, b), b in bounds; schur (if not null, stop_at > 0) receives the
    // (n - stop_at)^2 skew Schur complement R + Q^T P^-1 Q, row-major with both triangles. A zero pivot at step k gives Pf 0
    // for the leading blocks up to k's segment end and zero with status invalid beyond it.
    PfaffianInfo pfaffian(int bits,const void* A,std::size_t n,void* out,const EliminationOptions& options={},void* leading=nullptr,void* schur=nullptr);
    // Complex skew-symmetric A (Complex<bits/32>; A_ij = -A_ji, no conjugation): the same sequence with complex steps; writes one
    // Complex<bits/32> to out (the product rounded once per component after each multiplication).
    PfaffianInfo pfaffian_complex(int bits,const void* A,std::size_t n,void* out,const EliminationOptions& options={},void* leading=nullptr,void* schur=nullptr);
    // Householder QR of A (m x n, row-major, m >= n); A is only read. info().rank reports rank deficiency.
    QRFactor factor_qr(int bits,const void* A,std::size_t m,std::size_t n,const QROptions& options={});
    // QR of the augmented Levenberg-Marquardt matrix [J; diag(d)] ((m+n) x n) for J (m x n) and the caller's diagonal d
    // (n entries, e.g. sqrt(mu) D): the same factorization of the stacked matrix. Solve with B of m+n rows ([r; 0]).
    QRFactor factor_qr_augmented(int bits,const void* J,std::size_t m,std::size_t n,const void* d,const QROptions& options={});
    // Complex Householder QR of A (m x n, row-major Complex<bits/32>, m >= n); see ComplexQRFactor below.
    ComplexQRFactor factor_qr_complex(int bits,const void* A,std::size_t m,std::size_t n,const QROptions& options={});
    const LinalgReport& report() const;
    const LinalgOptions& options() const;
    // Retained memory, and its release: the scratch and the idle resident workspaces are freed now; busy resident workspaces are
    // detached and stay valid for the batches holding them, which free them when their submission's completion step has run (wait
    // or destruction of the submission) or the unsubmitted batch is destroyed. Later calls allocate afresh (a batch that already
    // holds a workspace gets a new one for its later calls). Compiled pipelines and worker threads are kept. Call on the Linalg's
    // host thread between its calls (not while a factorization of that thread runs); other threads may keep encoding resident
    // products. Returns what was freed (scratch, idle) and detached (busy).
    LinalgWorkspaces workspaces() const;
    LinalgWorkspaces release_workspaces();

    // ---- Resident products (docs/numerics.md, "Resident products"; docs/execution.md, "Resident units") ----
    // syrk / gemm encoded into an Engine's CommandBatch on Buffer<Float<bits>> operands of that engine (row-major, sizes as the
    // host calls; buffers may be larger). After Submission::wait() C equals the host-array call bit for bit. Band analysis,
    // member lists and the modulus count are GPU passes of the batch; outputs of host-fallback lines are computed at wait, so C
    // is provisional within the batch (later writes rejected, reads flagged). C must not be A or B. report() is not changed.
    template<class T> LinalgTicket syrk(CommandBatch& batch,const Buffer<T>& A,std::size_t rows,std::size_t cols,Buffer<T>& C,bool lower_only=true,bool subtract=false){
        return encode_product(batch,detail::real_bits<T>(),true,false,detail::Access::operand(A),detail::Access::operand(A),cols,cols,rows,detail::Access::operand(C),lower_only,subtract);}
    template<class T> LinalgTicket gemm(CommandBatch& batch,bool transpose_a,const Buffer<T>& A,const Buffer<T>& B,std::size_t m,std::size_t n,std::size_t k,Buffer<T>& C,bool subtract=false){
        return encode_product(batch,detail::real_bits<T>(),false,transpose_a,detail::Access::operand(A),detail::Access::operand(B),m,n,k,detail::Access::operand(C),false,subtract);}
    // GemmSpec form on Buffer operands (offsets and leading dimensions address submatrices of the buffers). The resident rules
    // above apply unchanged: A, B and C are distinct buffers, C is provisional until wait, and require_gpu throws invalid_argument.
    template<class T> LinalgTicket gemm(CommandBatch& batch,const Buffer<T>& A,const Buffer<T>& B,std::size_t m,std::size_t n,std::size_t k,Buffer<T>& C,const GemmSpec& spec){
        return encode_product(batch,detail::real_bits<T>(),false,spec.transpose_a,detail::Access::operand(A),detail::Access::operand(B),m,n,k,detail::Access::operand(C),false,
                              spec.update!=GemmUpdate::overwrite,&spec);}
    // Factorizations and solves on Buffer operands: synchronous, outside any batch (the host panels run between dependent GPU
    // updates). The buffers must be idle; the call owns them like a submission until it returns (concurrent mapping or
    // submission throws logic_error), the GPU uses them in place (no copies) and the results equal the host-array calls bit for bit.
    template<class T> CholeskyInfo cholesky(const Buffer<T>& A,std::size_t n,Buffer<T>& L,const FactorOptions& options={}){
        return cholesky_buffers(detail::real_bits<T>(),detail::Access::operand(A),n,detail::Access::operand(L),options);}
    template<class T> Timing trsm(bool transpose,const Buffer<T>& L,std::size_t n,const Buffer<T>& B,std::size_t nrhs,Buffer<T>& X,const FactorOptions& options={}){
        return solve_buffers(detail::real_bits<T>(),transpose?2:1,detail::Access::operand(L),n,detail::Access::operand(B),nrhs,detail::Access::operand(X),options);}
    template<class T> Timing cholesky_solve(const Buffer<T>& L,std::size_t n,const Buffer<T>& B,std::size_t nrhs,Buffer<T>& X,const FactorOptions& options={}){
        return solve_buffers(detail::real_bits<T>(),3,detail::Access::operand(L),n,detail::Access::operand(B),nrhs,detail::Access::operand(X),options);}
    template<class T> QRFactor factor_qr(const Buffer<T>& A,std::size_t m,std::size_t n,const QROptions& options={}){
        return factor_qr_buffer(detail::real_bits<T>(),detail::Access::operand(A),m,n,options);}
    // Explicit contract names. Existing entry points keep identical defaults and behavior.
    Timing gemm_exact(int bits,bool transpose_a,const void* A,const void* B,std::size_t m,std::size_t n,std::size_t k,void* C,bool subtract=false){return gemm(bits,transpose_a,A,B,m,n,k,C,subtract);}
    Timing syrk_exact(int bits,const void* A,std::size_t rows,std::size_t cols,void* C,bool lower_only=true,bool subtract=false){return syrk(bits,A,rows,cols,C,lower_only,subtract);}
    template<class T> LinalgTicket gemm_exact(CommandBatch& b,bool transpose_a,const Buffer<T>& A,const Buffer<T>& B,std::size_t m,std::size_t n,std::size_t k,Buffer<T>& C,bool subtract=false){return gemm(b,transpose_a,A,B,m,n,k,C,subtract);}
    template<class T> LinalgTicket syrk_exact(CommandBatch& b,const Buffer<T>& A,std::size_t rows,std::size_t cols,Buffer<T>& C,bool lower_only=true,bool subtract=false){return syrk(b,A,rows,cols,C,lower_only,subtract);}
    CholeskyInfo cholesky_blocked(int bits,const void* A,std::size_t n,void* L,const FactorOptions& o={}){return cholesky(bits,A,n,L,o);}
    Timing cholesky_solve_blocked(int bits,const void* L,std::size_t n,const void* B,std::size_t nrhs,void* X,const FactorOptions& o={}){return cholesky_solve(bits,L,n,B,nrhs,X,o);}
    template<class T> CholeskyInfo cholesky_blocked(const Buffer<T>& A,std::size_t n,Buffer<T>& L,const FactorOptions& o={}){return cholesky(A,n,L,o);}
    template<class T> Timing cholesky_solve_blocked(const Buffer<T>& L,std::size_t n,const Buffer<T>& B,std::size_t nrhs,Buffer<T>& X,const FactorOptions& o={}){return cholesky_solve(L,n,B,nrhs,X,o);}
private:
    LinalgTicket encode_product(CommandBatch&,int bits,bool syrk,bool transpose_a,const detail::Operand& A,const detail::Operand& B,std::size_t m,std::size_t n,
                                std::size_t k,const detail::Operand& C,bool lower_only,bool subtract,const GemmSpec* spec=nullptr);
    CholeskyInfo cholesky_buffers(int bits,const detail::Operand& A,std::size_t n,const detail::Operand& L,const FactorOptions&);
    Timing solve_buffers(int bits,int passes,const detail::Operand& L,std::size_t n,const detail::Operand& B,std::size_t nrhs,const detail::Operand& X,const FactorOptions&);
    QRFactor factor_qr_buffer(int bits,const detail::Operand& A,std::size_t m,std::size_t n,const QROptions&);
    friend class QRFactor; friend class LUFactor; friend struct detail::LinalgHooks; struct Impl; std::unique_ptr<Impl> impl;
};
// ---- Exact dot product on the CPU: RN(sum_k a[k*sa]*b[k*sb]) with one rounding (the GPU fallback) ----
namespace detail {
// Rounds sign * integer(w) * 2^scale to N words exactly as core.hpp pack() does, for any word count.
template<int N> Number<N> pack_words(const word* w,std::size_t count,exponent_type scale,int sign,word status){
    std::size_t top=count;while(top&&!w[top-1])--top;if(!top)return zero<N>(status);
    long h=long(32*(top-1))+31-__builtin_clz(w[top-1]);
    exponent_type e=scale+h;if(e>1000000000||e< -1000000001)return zero<N>(status|exponent_overflow);
    // 32 bits of w from bit p (bits outside w are zero).
    auto chunk=[&](long p)->word{if(p<=-32)return 0;if(p<0)return w[0]<<(-p);std::size_t i=std::size_t(p/32);int s=int(p%32);
        word x=i<count?w[i]>>s:0;if(s&&i+1<count)x|=w[i+1]<<(32-s);return x;};
    Number<N> r=zero<N>(status);r.sign=sign;r.exponent=int(e);long shift=h-(32*N-1);
    for(int i=0;i<N;++i)r.limb[i]=chunk(shift+32L*i);
    if(shift<=0)return checked(r);
    long rb=shift-1;bool round=(w[rb/32]>>(rb%32))&1,below=(rb%32)&&(w[rb/32]&((word(1)<<(rb%32))-1));
    for(long i=0;i<rb/32&&!below;++i)below=w[i]!=0;
    if(round&&(below||(r.limb[0]&1)))increment(r);
    return checked(r);
}
template<int N> Number<N> pack_words(const std::vector<word>& w,exponent_type scale,int sign,word status){return pack_words<N>(w.data(),w.size(),scale,sign,status);}
inline std::uint64_t load64(const word* p){std::uint64_t v;std::memcpy(&v,p,8);return v;}
inline void store64(word* p,std::uint64_t v){std::memcpy(p,&v,8);}
// acc += x * 2^shift for a nonnegative accumulator wide enough by construction (64-bit steps over pairs of words).
inline void add_shifted(word* acc,const word* x,int words,long shift){
    std::size_t w=std::size_t(shift/32);const int s=int(shift%32);unsigned __int128 carry=0;int i=0;std::uint64_t prev=0;
    for(;i+2<=words;i+=2){const std::uint64_t X=load64(x+i),v=(X<<s)|(s?prev>>(64-s):0);prev=X;
        carry+=(unsigned __int128)load64(acc+w+i)+v;store64(acc+w+i,std::uint64_t(carry));carry>>=64;}
    dword c=dword(carry);
    for(;i<=words;++i){word v=i<words?x[i]<<s:0;if(s&&i>0)v|=x[i-1]>>(32-s);dword t=dword(acc[w+i])+v+c;acc[w+i]=word(t);c=t>>32;}
    for(std::size_t k=w+std::size_t(words)+1;c;++k){dword t=dword(acc[k])+c;acc[k]=word(t);c=t>>32;}
}
// p (2N words) = x * y exactly for N-word mantissas, with 64-bit limbs (N may be odd).
template<int N> inline void mantissa_mul(const word* x,const word* y,word* p){
    constexpr int H=(N+1)/2;std::uint64_t a[H],b[H],r[2*H];
    for(int i=0;i<H;++i){a[i]=x[2*i];b[i]=y[2*i];if(2*i+1<N){a[i]|=std::uint64_t(x[2*i+1])<<32;b[i]|=std::uint64_t(y[2*i+1])<<32;}}
    for(int i=0;i<2*H;++i)r[i]=0;
    for(int i=0;i<H;++i){unsigned __int128 cy=0;for(int j=0;j<H;++j){cy+=(unsigned __int128)a[i]*b[j]+r[i+j];r[i+j]=std::uint64_t(cy);cy>>=64;}r[i+H]=std::uint64_t(cy);}
    for(int i=0;i<N;++i){p[2*i]=word(r[i]);p[2*i+1]=word(r[i]>>32);}
}
// Two's complement accumulator: acc += sign * x * 2^shift (acc wide enough by construction).
inline void accumulate(std::vector<word>& acc,const word* x,int words,long shift,int sign){
    std::size_t w=std::size_t(shift/32);int s=int(shift%32);dword carry=0;bool borrow=false;
    for(std::size_t i=0;i+w<acc.size();++i){
        word v=0;if(i<std::size_t(words))v=x[i]<<s;if(s&&i>0&&i-1<std::size_t(words))v|=x[i-1]>>(32-s);
        if(i>std::size_t(words)&&!carry&&!borrow)break;
        if(sign>0){dword t=dword(acc[i+w])+v+carry;acc[i+w]=word(t);carry=t>>32;}
        else{dword t=dword(v)+borrow;borrow=dword(acc[i+w])<t;acc[i+w]=word(dword(acc[i+w])-t);}
    }
}
}
// RN(c - sum_k a[k*sa]*b[k*sb]) (subtract) or RN(c + sum ...) with one rounding; c == nullptr is a zero addend.
// The addend is one more exact term. Terms are sorted by exponent and grouped into clusters; a new cluster starts
// when every remaining term lies more than bits+log2(K+1)+4 bits below the current cluster's lowest bit. Each
// cluster is summed exactly. With S the first nonzero cluster sum and F the rest, |F| < 2^(lsb(S)-bits-2), so
// RN(S+F) = RN(S + sign(F)*eps) and sign(F) is the sign of the next nonzero cluster sum. Statuses: zero with the OR
// of the statuses of c and of every a[k], b[k]; without nonzero products the result is c itself.
template<int N> Number<N> exact_dot_add(const Number<N>* c,bool subtract,const Number<N>* a,std::ptrdiff_t sa,const Number<N>* b,std::ptrdiff_t sb,std::size_t K){
    word status=c?c->status:0;for(std::size_t k=0;k<K;++k)status|=a[std::ptrdiff_t(k)*sa].status|b[std::ptrdiff_t(k)*sb].status;
    if(status)return zero<N>(status);
    struct Term{exponent_type scale;std::size_t k;};Term small[65];std::vector<Term> big;Term* t=small;std::size_t count=0;
    if(K+1>65){big.resize(K+1);t=big.data();}
    for(std::size_t k=0;k<K;++k){const auto &x=a[std::ptrdiff_t(k)*sa],&y=b[std::ptrdiff_t(k)*sb];
        if(x.sign&&y.sign)t[count++]={exponent_type(x.exponent)+y.exponent-2*(32*N-1),k};}
    if(!count)return c?*c:zero<N>();
    if(c&&c->sign)t[count++]={exponent_type(c->exponent)-(32*N-1),K}; // the addend, an N-word term, is k = K
    // Exact term k (2N words, or the N-word addend) into p; returns its sign.
    auto term=[&](std::size_t k,word* p,int& words)->int{
        if(k==K){for(int u=0;u<N;++u)p[u]=c->limb[u];words=N;return c->sign;}
        const auto &x=a[std::ptrdiff_t(k)*sa],&y=b[std::ptrdiff_t(k)*sb];detail::mantissa_mul<N>(x.limb,y.limb,p);words=2*N;
        return subtract?-x.sign*y.sign:x.sign*y.sign;};
    // Common case: all terms fit one exact accumulator pair (positive and negative parts), one rounding of the difference.
    {exponent_type hi=t[0].scale,lo=t[0].scale;for(std::size_t i=1;i<count;++i){hi=std::max(hi,t[i].scale);lo=std::min(lo,t[i].scale);}
     int logc=0;while((std::size_t(1)<<logc)<count)++logc;constexpr std::size_t cap=256;const exponent_type need=hi-lo+64*N+logc+1;
     if(need<=32*exponent_type(cap-2)){const std::size_t W=std::size_t(need/32)+2;word pos[cap],neg[cap],p[2*N];std::fill(pos,pos+W,0);std::fill(neg,neg+W,0);
        for(std::size_t q=0;q<count;++q){int words;int sign=term(t[q].k,p,words);detail::add_shifted(sign>0?pos:neg,p,words,long(t[q].scale-lo));}
        int cmp=0;for(std::size_t i=W;i-->0;)if(pos[i]!=neg[i]){cmp=pos[i]>neg[i]?1:-1;break;}
        if(!cmp)return zero<N>();
        word* x=cmp>0?pos:neg;const word* y=cmp>0?neg:pos;dword borrow=0;
        for(std::size_t i=0;i<W;++i){dword v=dword(y[i])+borrow;borrow=dword(x[i])<v;x[i]=word(dword(x[i])-v);}
        return detail::pack_words<N>(x,W,lo,cmp,ok);}}
    std::sort(t,t+count,[](const Term& u,const Term& v){return u.scale>v.scale;});
    int logk=0;while((std::size_t(1)<<logk)<K+1)++logk;
    const exponent_type gap=32*N+logk+4,width=64*N; // a term is < 2^(scale+64N)
    struct Sum{int sign;exponent_type scale;std::vector<word> mag;};std::vector<Sum> sums;
    for(std::size_t i=0;i<count&&sums.size()<2;){
        std::size_t j=i+1;exponent_type low=t[i].scale;
        while(j<count&&t[j].scale+width-1+gap>=low){low=std::min(low,t[j].scale);++j;}
        std::vector<word> acc(std::size_t((t[i].scale+width-low)/32+3),0);word p[2*N];
        for(std::size_t q=i;q<j;++q){int words;int sign=term(t[q].k,p,words);detail::accumulate(acc,p,words,long(t[q].scale-low),sign);}
        int sign=1;if(acc.back()>>31){sign=-1;dword cy=1;for(auto& v:acc){dword z=dword(word(~v))+cy;v=word(z);cy=z>>32;}}
        if(std::any_of(acc.begin(),acc.end(),[](word v){return v!=0;}))sums.push_back({sign,low,std::move(acc)});
        i=j;
    }
    if(sums.empty())return zero<N>();
    const Sum& s=sums[0];if(sums.size()==1)return detail::pack_words<N>(s.mag,s.scale,s.sign,ok);
    // S*2^D -/+ 1: D puts eps below the rounding position even when S is narrower than the target.
    const int D=32*N+64;std::vector<word> ext(s.mag.size()+D/32+1,0);
    for(std::size_t i=0;i<s.mag.size();++i)ext[i+D/32]=s.mag[i];
    if(sums[1].sign==s.sign)ext[0]=1;else for(auto& v:ext){word old=v;v=old-1;if(old)break;}
    return detail::pack_words<N>(ext,s.scale-D,s.sign,ok);
}
// RN(sum_k a[k*sa]*b[k*sb]) with one rounding (the GPU fallback); empty K and exact cancellation give canonical zero.
template<int N> Number<N> exact_dot(const Number<N>* a,std::ptrdiff_t sa,const Number<N>* b,std::ptrdiff_t sb,std::size_t K){
    return exact_dot_add<N>(nullptr,false,a,sa,b,sb,K);}
// ---- Complex Householder QR (round 43; docs/numerics.md, "Complex QR factorization and least squares") ----
namespace detail { struct ComplexQRData; }
// Owning QR factor of a complex m x n matrix A (m >= n), from Linalg::factor_qr_complex: A = Q R with Q = H_0 ... H_{n-1},
// H_j = I - tau_j v_j v_j^H and complex tau_j (the LAPACK zgeqrf convention), so H_j^H maps the reduced column (alpha; x) to
// (beta_j; 0) with real beta_j = -sgn(Re alpha) ||(alpha; x)|| (sgn(0) = +1): R is upper triangular with a real diagonal.
// Options, rank and status rules, ownership and threading are those of QRFactor (block, rank_bits, GPU/host split, which never
// changes a bit). QROptions::pivot: column pivoting as the real QR (xGEQP3 style; real squared column norms downdated by |r_jk|^2,
// exact comparisons, ties to the lowest original index): A P = Q R, every array below belongs to A P, and a rank-deficient
// pivoted factor solves for the basic (not minimum-norm) solution.
class ComplexQRFactor {
public:
    ComplexQRFactor(); ~ComplexQRFactor(); ComplexQRFactor(ComplexQRFactor&&) noexcept; ComplexQRFactor& operator=(ComplexQRFactor&&) noexcept;
    bool empty() const; int bits() const; std::size_t rows() const; std::size_t cols() const; std::size_t block() const;
    const QRInfo& info() const; bool full_rank() const;
    // Row-major Complex<bits/32> arrays owned by the factor: R (n x n, upper, real diagonal; with rank p < n, rows i >= p hold
    // the unreduced remainder in columns j >= p), V (m x n: column j is v_j, zero above row j, v_j[j] = the scaled v0), tau (n)
    // and T (blocks of nb x nb, the compact-WY upper triangles: H_k0 ... H_k1-1 = I - V_b T_b V_b^H; block b at offset b*nb*nb).
    const void* r() const; const void* v() const; const void* tau() const; const void* t() const;
    bool pivoted() const; const std::size_t* permutation() const;  // column j of R and V is column permutation()[j] of A (identity unpivoted)
    // X (n x nrhs, complex) = least-squares solution of min ||A x - b|| for each column b of B (m x nrhs): Q^H B, then
    // R X = (Q^H B)[0:n) (pivoted: x[perm[i]] = z_i). Rank deficient (rank p < n): an unpivoted factor writes zero with status
    // invalid to every entry of X; a pivoted one gives the basic solution x[perm[i]] = z_i from R[0:p,0:p] z = (Q^H b)[0:p) and
    // x[perm[i]] = 0 for i >= p; a status column gives zero with that status (and invalid) everywhere. X must not overlap B.
    Timing solve(const void* B,std::size_t nrhs,void* X) const;
    // B (m x nrhs, complex) <- Q^H B (adjoint) or Q B, in place.
    Timing apply_q(void* B,std::size_t nrhs,bool adjoint) const;
private:
    friend class Linalg; std::unique_ptr<detail::ComplexQRData> d;
};
}
