module MatrixCoversSparseArraysUnitfulExt

# `MatrixCoversSparseArraysExt` types the matrix slot of its refiners, where `MatrixCoversUnitfulExt` types
# the element: neither is more specific for a sparse matrix of quantities, so the two
# are ambiguous there. These methods resolve that pair. They are the only overlap --
# every other sparse method leaves its matrix slot untyped.

using LinearAlgebra: LinearAlgebra, Hermitian, Symmetric
using MatrixCovers
using MatrixCovers: AbsLog
using SparseArrays: SparseArrays, SparseMatrixCSC
using Unitful: Quantity

const MC = MatrixCovers

# Sparse storage synthesizes structural zeros with `zero(eltype)`, so the element
# type is concrete and every entry carries the same unit.
const QSparse = SparseMatrixCSC{<:Quantity}
const QSparseSym = Union{QSparse,
                         Symmetric{<:Quantity,<:SparseMatrixCSC},
                         Hermitian{<:Quantity,<:SparseMatrixCSC}}

# `Unitful` triggers `MatrixCoversUnitfulExt` as well as this extension, so it is loaded whenever
# these methods can be called. Reaching for it here rather than at load time leaves
# the two extensions' load order free.
unitfulext() = Base.get_extension(MatrixCovers, :MatrixCoversUnitfulExt)::Module

MC.symcover_min!(ϕ::AbsLog{2}, a::AbstractVector{<:Quantity}, A::QSparseSym; kwargs...) =
    unitfulext().symstart!(MC.symcover_min!, a, A, ϕ; kwargs...)

MC.cover_min!(ϕ::AbsLog{2}, a::AbstractVector{<:Quantity}, b::AbstractVector{<:Quantity},
               A::QSparse; kwargs...) =
    unitfulext().asymstart!(MC.cover_min!, a, b, A, ϕ; kwargs...)

end  # module MatrixCoversSparseArraysUnitfulExt
