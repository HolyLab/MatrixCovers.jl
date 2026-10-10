# Cholesky factorization of a symmetric positive definite matrix whose
# off-diagonal pattern is a forest. Eliminating each tree from its leaves
# produces no fill, so the factor `L` of `M = L Lᵀ` has one diagonal entry per
# vertex and one off-diagonal entry per non-root vertex, and factorization,
# solves, and products are linear in the number of vertices. The arithmetic is
# that of the element type, so any element type serves.
struct TreeCholesky{T}
    parent::Vector{Int}   # parent vertex; 0 for a root
    order::Vector{Int}    # the vertices, each after all of its children
    d::Vector{T}          # `L[v, v]`
    l::Vector{T}          # `L[parent[v], v]`; 0 for a root
end

# Factor the `N×N` matrix with diagonal `diag` and off-diagonal entries `V[k]`
# at `(I[k], J[k])` and `(J[k], I[k])`. The edges must form a forest, and the
# matrix must be positive definite; either failure throws.
function tree_cholesky(N::Int, I::AbstractVector{Int}, J::AbstractVector{Int},
                       V::AbstractVector{T}, diag::AbstractVector{T}) where {T}
    axes(I) == axes(J) == axes(V) ||
        throw(DimensionMismatch("edge endpoints and values must share axes, got $(axes(I)), $(axes(J)), $(axes(V))"))
    length(diag) == N || throw(DimensionMismatch("the diagonal has $(length(diag)) entries; expected $N"))
    ptr = zeros(Int, N + 1)
    for k in eachindex(I)
        ptr[I[k]+1] += 1
        ptr[J[k]+1] += 1
    end
    ptr[1] = 1
    cumsum!(ptr, ptr)
    adj = zeros(Int, ptr[end] - 1)
    cursor = ptr[1:N]
    for k in eachindex(I)
        adj[cursor[I[k]]] = k
        cursor[I[k]] += 1
        adj[cursor[J[k]]] = k
        cursor[J[k]] += 1
    end
    # Breadth-first search from each root lists parents before children.
    parent = zeros(Int, N)
    pedge = zeros(Int, N)
    seen = falses(N)
    bfs = Int[]
    sizehint!(bfs, N)
    for r in 1:N
        seen[r] && continue
        seen[r] = true
        head = length(bfs) + 1
        push!(bfs, r)
        while head <= length(bfs)
            v = bfs[head]
            head += 1
            for s in ptr[v]:ptr[v+1]-1
                k = adj[s]
                k == pedge[v] && continue
                w = I[k] == v ? J[k] : I[k]
                seen[w] && throw(ArgumentError("the edges contain a cycle through vertex $w"))
                seen[w] = true
                parent[w] = v
                pedge[w] = k
                push!(bfs, w)
            end
        end
    end
    order = reverse!(bfs)
    d = zeros(T, N)
    l = zeros(T, N)
    s = collect(T, diag)   # each pivot, less the squares of its children's factor entries
    for v in order
        s[v] > 0 || throw(ArgumentError("the matrix is not positive definite: pivot $(s[v]) at vertex $v"))
        d[v] = sqrt(s[v])
        p = parent[v]
        if p != 0
            l[v] = V[pedge[v]] / d[v]
            s[p] -= l[v]^2
        end
    end
    return TreeCholesky{T}(parent, order, d, l)
end

# `x = L \ b`; `x` may be `b`.
function solve_ptl!(x::AbstractVector, F::TreeCholesky, b::AbstractVector)
    x === b || copyto!(x, b)
    for v in F.order
        x[v] /= F.d[v]
        p = F.parent[v]
        p == 0 || (x[p] -= F.l[v] * x[v])
    end
    return x
end

# `x = Lᵀ \ b`; `x` may be `b`.
function solve_up!(x::AbstractVector, F::TreeCholesky, b::AbstractVector)
    x === b || copyto!(x, b)
    for v in Iterators.reverse(F.order)
        p = F.parent[v]
        p == 0 || (x[v] -= F.l[v] * x[p])
        x[v] /= F.d[v]
    end
    return x
end

# `y = Lᵀ x`; `y` may be `x`.
function lt_mul!(y::AbstractVector, F::TreeCholesky, x::AbstractVector)
    for v in F.order
        p = F.parent[v]
        y[v] = F.d[v] * x[v] + (p == 0 ? zero(eltype(y)) : F.l[v] * x[p])
    end
    return y
end

# Indices of the edges of a maximum-weight spanning forest of the graph on
# `1:N` with edges `(I[k], J[k])` and weights `W[k]`, by Kruskal's algorithm.
function max_weight_forest(N::Int, I::AbstractVector{Int}, J::AbstractVector{Int}, W::AbstractVector)
    axes(I) == axes(J) == axes(W) ||
        throw(DimensionMismatch("edge endpoints and weights must share axes, got $(axes(I)), $(axes(J)), $(axes(W))"))
    uf = collect(1:N)
    find(x) = (while uf[x] != x; uf[x] = uf[uf[x]]; x = uf[x]; end; x)
    keep = Int[]
    for k in sortperm(W; rev=true)
        rp, rq = find(I[k]), find(J[k])
        rp == rq && continue
        uf[rp] = rq
        push!(keep, k)
    end
    return keep
end
