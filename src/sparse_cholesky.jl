# Sparse Cholesky factorization through CHOLMOD with a locally owned workspace.
#
# `SparseArrays.CHOLMOD` keeps its `cholmod_common` in task-local storage
# behind an abstract `Ref`, so every call through it is a dynamic dispatch and
# `juliac --trim=safe` rejects code that reaches it. This driver owns one
# concretely typed workspace per factorization object and calls the same
# SuiteSparse routines with the same parameters (`AMD` ordering with
# postordering, `LL'` output), so the factors and solves are identical to those
# of `cholesky(Symmetric(A, :U))`. Only `Float64` values and `Int` indices are
# supported, which is all the solvers in this package factor.

using SparseArrays.LibSuiteSparse: LibSuiteSparse, cholmod_common, cholmod_sparse, cholmod_dense,
    cholmod_factor, cholmod_l_start, cholmod_l_finish, cholmod_l_analyze, cholmod_l_factorize,
    cholmod_l_solve2, cholmod_l_free_factor, cholmod_l_free_dense,
    CHOLMOD_REAL, CHOLMOD_DOUBLE, CHOLMOD_LONG, CHOLMOD_OK, CHOLMOD_NOT_POSDEF,
    CHOLMOD_A, CHOLMOD_L, CHOLMOD_Lt, CHOLMOD_P, CHOLMOD_Pt

# SuiteSparse's allocator hooks are process-global. Route them through Julia's
# counted allocator, as `SparseArrays` does, so that memory freed here matches
# memory allocated after `SparseArrays` initializes CHOLMOD, and vice versa.
const _SUITESPARSE_LOCK = ReentrantLock()
const _SUITESPARSE_READY = Ref(false)
function _suitesparse_init()
    _SUITESPARSE_READY[] && return nothing
    lock(_SUITESPARSE_LOCK) do
        _SUITESPARSE_READY[] && return nothing
        LibSuiteSparse.SuiteSparse_config_malloc_func_set(cglobal(:jl_malloc, Ptr{Cvoid}))
        LibSuiteSparse.SuiteSparse_config_calloc_func_set(cglobal(:jl_calloc, Ptr{Cvoid}))
        LibSuiteSparse.SuiteSparse_config_realloc_func_set(cglobal(:jl_realloc, Ptr{Cvoid}))
        LibSuiteSparse.SuiteSparse_config_free_func_set(cglobal(:jl_free, Ptr{Cvoid}))
        _SUITESPARSE_READY[] = true
        return nothing
    end
    return nothing
end

# A symmetric positive-definite factorization `P*A*P' = L*L'` of the upper
# triangle of a sparse matrix. `analyze!` fixes the pattern and ordering;
# `factorize!` fills in the values; `solve!` applies `A`, `L`, or `P` pieces.
mutable struct SparseCholesky
    common::cholmod_common
    A::cholmod_sparse              # header over `colptr`, `rowval`, and the caller's values
    colptr::Vector{Int}            # zero-based copies of the analyzed pattern
    rowval::Vector{Int}
    B::cholmod_dense               # header over a right-hand side
    L::Ptr{cholmod_factor}
    X::Base.RefValue{Ptr{cholmod_dense}}   # solve workspaces CHOLMOD grows as needed
    Y::Base.RefValue{Ptr{cholmod_dense}}
    E::Base.RefValue{Ptr{cholmod_dense}}
    n::Int

    function SparseCholesky()
        _suitesparse_init()
        common = cholmod_common()
        GC.@preserve common begin
            cholmod_l_start(Ptr{cholmod_common}(pointer_from_objref(common))) == LibSuiteSparse.TRUE ||
                error("cholmod_l_start failed")
        end
        common.print = 0            # errors are reported through `status`
        common.nmethods = 2         # user permutation (none given), then AMD; no METIS
        common.postorder = 1
        common.final_ll = 1
        F = new(common, cholmod_sparse(), Int[], Int[], cholmod_dense(), C_NULL,
                Ref(Ptr{cholmod_dense}(C_NULL)), Ref(Ptr{cholmod_dense}(C_NULL)),
                Ref(Ptr{cholmod_dense}(C_NULL)), 0)
        return finalizer(_free!, F)
    end
end

_common(F::SparseCholesky) = Ptr{cholmod_common}(pointer_from_objref(F.common))

function _free!(F::SparseCholesky)
    GC.@preserve F begin
        common = _common(F)
        F.L == C_NULL || cholmod_l_free_factor(Ref(F.L), common)
        F.L = C_NULL
        for W in (F.X, F.Y, F.E)
            W[] == C_NULL || cholmod_l_free_dense(W, common)
        end
        cholmod_l_finish(common)
    end
    return nothing
end

function _check_status(F::SparseCholesky, what::String)
    status = F.common.status
    status == CHOLMOD_OK && return nothing
    status == CHOLMOD_NOT_POSDEF && throw(LinearAlgebra.PosDefException(1))
    error("CHOLMOD failed to $what (status $status)")
end

# Point the sparse header at the analyzed pattern and the values of `S`.
function _set_sparse!(F::SparseCholesky, S::SparseMatrixCSC{Float64,Int})
    n = F.n
    size(S) == (n, n) || throw(DimensionMismatch("expected a $n×$n matrix, got size $(size(S))"))
    nz = length(F.rowval)
    length(nonzeros(S)) == nz || throw(ArgumentError("the pattern of `S` differs from the analyzed pattern"))
    A = F.A
    A.nrow = n
    A.ncol = n
    A.nzmax = nz
    A.p = pointer(F.colptr)
    A.i = pointer(F.rowval)
    A.nz = C_NULL
    A.x = pointer(nonzeros(S))
    A.z = C_NULL
    A.stype = 1                    # upper triangle stored; the lower triangle is ignored
    A.itype = CHOLMOD_LONG
    A.xtype = CHOLMOD_REAL
    A.dtype = CHOLMOD_DOUBLE
    A.sorted = 1
    A.packed = 1
    return A
end

# Symbolic analysis of the upper triangle of `S`. The rows within each column
# of `S` must be sorted.
function analyze!(F::SparseCholesky, S::SparseMatrixCSC{Float64,Int})
    n = size(S, 1)
    size(S, 2) == n || throw(DimensionMismatch("the matrix must be square, got size $(size(S))"))
    F.n = n
    resize!(F.colptr, n + 1)
    resize!(F.rowval, nnz(S))
    F.colptr .= SparseArrays.getcolptr(S) .- 1
    F.rowval .= rowvals(S) .- 1
    A = _set_sparse!(F, S)
    GC.@preserve S F begin
        common = _common(F)
        F.L == C_NULL || cholmod_l_free_factor(Ref(F.L), common)
        F.L = C_NULL
        F.L = cholmod_l_analyze(Ptr{cholmod_sparse}(pointer_from_objref(A)), common)
        if F.L == C_NULL
            _check_status(F, "analyze the pattern")
            error("CHOLMOD returned no symbolic factorization")
        end
    end
    return F
end

# Numeric factorization of `S`, whose pattern must be the analyzed one.
function factorize!(F::SparseCholesky, S::SparseMatrixCSC{Float64,Int})
    F.L == C_NULL && throw(ArgumentError("`analyze!` must run before `factorize!`"))
    A = _set_sparse!(F, S)
    GC.@preserve S F begin
        cholmod_l_factorize(Ptr{cholmod_sparse}(pointer_from_objref(A)), F.L, _common(F))
        _check_status(F, "factorize the matrix")
    end
    return F
end

# Number of stored values in the factor `analyze!` predicts, for a fill budget.
function factor_entries(F::SparseCholesky)
    F.L == C_NULL && throw(ArgumentError("`analyze!` must run before `factor_entries`"))
    s = unsafe_load(F.L)
    Int(s.n) == F.n || error("CHOLMOD analyzed a matrix of order $(Int(s.n)), but the driver has order $(F.n)")
    s.is_super == 0 || return Int(s.xsize)
    counts = unsafe_wrap(Array, convert(Ptr{Int}, s.ColCount), Int(s.n))
    return sum(Int, counts)
end

# `X = op \ B` for the system `sys` (`CHOLMOD_A`, `CHOLMOD_L`, `CHOLMOD_Lt`,
# `CHOLMOD_P`, or `CHOLMOD_Pt`). `X` and `B` may be the same array.
function solve!(X::StridedVecOrMat{Float64}, F::SparseCholesky, sys::Integer, B::StridedVecOrMat{Float64})
    F.L == C_NULL && throw(ArgumentError("`factorize!` must run before `solve!`"))
    n = F.n
    size(B, 1) == n || throw(DimensionMismatch("the right-hand side has $(size(B, 1)) rows; expected $n"))
    size(X) == size(B) || throw(DimensionMismatch("solution size $(size(X)) does not match right-hand side size $(size(B))"))
    (stride(B, 1) == 1 && stride(X, 1) == 1) || throw(ArgumentError("columns must be contiguous"))
    k = size(B, 2)
    D = F.B
    D.nrow = n
    D.ncol = k
    D.d = k > 1 ? stride(B, 2) : n
    D.nzmax = D.d * k
    D.x = pointer(B)
    D.z = C_NULL
    D.xtype = CHOLMOD_REAL
    D.dtype = CHOLMOD_DOUBLE
    GC.@preserve B F begin
        ok = cholmod_l_solve2(sys, F.L, Ptr{cholmod_dense}(pointer_from_objref(D)), C_NULL,
                              F.X, C_NULL, F.Y, F.E, _common(F))
        ok == LibSuiteSparse.TRUE || _check_status(F, "solve")
        xs = unsafe_load(F.X[])
        (Int(xs.nrow) == n && Int(xs.ncol) == k) ||
            error("CHOLMOD returned a $(Int(xs.nrow))×$(Int(xs.ncol)) solution; expected $n×$k")
        src = convert(Ptr{Float64}, xs.x)
        ld = Int(xs.d)
        for j in 1:k
            unsafe_copyto!(pointer(X, (j - 1) * n + 1), src + (j - 1) * ld * sizeof(Float64), n)
        end
    end
    return X
end

# `X = (P'L) \ B` and `X = (L'P) \ B`, the two halves of `A = (P'L)(L'P)`.
function solve_ptl!(X, F::SparseCholesky, B)
    solve!(X, F, CHOLMOD_P, B)     # CHOLMOD_P applies the permutation `x = P*b`
    return solve!(X, F, CHOLMOD_L, X)
end
function solve_up!(X, F::SparseCholesky, B)
    solve!(X, F, CHOLMOD_Lt, B)
    return solve!(X, F, CHOLMOD_Pt, X)
end
