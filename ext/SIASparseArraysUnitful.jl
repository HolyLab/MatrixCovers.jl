module SIASparseArraysUnitful

# `SIASparseArrays` types the matrix slot of its refiners, where `SIAUnitful` types
# the element: neither is more specific for a sparse matrix of quantities, so the two
# are ambiguous there. These methods resolve that pair. They are the only overlap --
# every other sparse method leaves its matrix slot untyped.

using LinearAlgebra
using ScaleInvariantAnalysis
using ScaleInvariantAnalysis: AbsLog
using SparseArrays
using Unitful: Quantity

const SIA = ScaleInvariantAnalysis

# Sparse storage synthesizes structural zeros with `zero(eltype)`, so the element
# type is concrete and every entry carries the same unit.
const QSparse = SparseMatrixCSC{<:Quantity}
const QSparseSym = Union{QSparse,
                         Symmetric{<:Quantity,<:SparseMatrixCSC},
                         Hermitian{<:Quantity,<:SparseMatrixCSC}}

# `Unitful` triggers `SIAUnitful` as well as this extension, so it is loaded whenever
# these methods can be called. Reaching for it here rather than at load time leaves
# the two extensions' load order free.
siaunitful() = Base.get_extension(ScaleInvariantAnalysis, :SIAUnitful)::Module

SIA.symcover_min!(ϕ::AbsLog{2}, a::AbstractVector{<:Quantity}, A::QSparseSym; kwargs...) =
    siaunitful().symstart!(SIA.symcover_min!, a, A, ϕ; kwargs...)

SIA.cover_min!(ϕ::AbsLog{2}, a::AbstractVector{<:Quantity}, b::AbstractVector{<:Quantity},
               A::QSparse; kwargs...) =
    siaunitful().asymstart!(SIA.cover_min!, a, b, A, ϕ; kwargs...)

end  # module SIASparseArraysUnitful
