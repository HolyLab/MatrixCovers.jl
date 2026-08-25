# SIMD-friendly logarithm using exponent reduction and an `atanh` series.
# Returned covers are certified separately.
#
# Callers pass nonnegative values. Zero, infinity, and NaN match `Base.log`.
@inline function _fastlog(x::Float64)
    # Scale subnormals into the normal range so exponent extraction sees them.
    sub = x < floatmin(Float64)
    xs = ifelse(sub, x * 0x1p54, x)
    bits = reinterpret(UInt64, xs)
    e = Float64(Int64(bits >> 52) - 1023) - ifelse(sub, 54.0, 0.0)
    m = reinterpret(Float64, (bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000)
    # Reduce the mantissa from [1, 2) to [√2/2, √2), centering log(m) on zero.
    big = m > 1.4142135623730951
    m = ifelse(big, 0.5 * m, m)
    e = ifelse(big, e + 1.0, e)
    f = m - 1.0
    s = f / (2.0 + f)
    # Six terms bound the truncation error below 6e-11.
    p = @evalpoly(s * s, 2.0, 2 / 3, 2 / 5, 2 / 7, 2 / 9, 2 / 11)
    r = e * 0.6931471805599453 + s * p
    return ifelse(iszero(x), -Inf, ifelse(x < Inf, r, x))
end
_fastlog(x::Float32) = Float32(_fastlog(Float64(x)))
_fastlog(x::Real) = log(x)   # element types the kernel does not cover

function _fastlog!(x::AbstractArray)
    @simd for k in eachindex(x)
        x[k] = _fastlog(x[k])
    end
    return x
end
