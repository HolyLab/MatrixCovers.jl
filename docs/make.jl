using MatrixCovers
using Documenter
using JuMP, HiGHS
using Unitful   # loaded here so the doctests' own `using` cannot emit precompilation output

DocMeta.setdocmeta!(MatrixCovers, :DocTestSetup, :(using MatrixCovers); recursive=true)

makedocs(;
    modules=[MatrixCovers],
    authors="Tim Holy <tim.holy@gmail.com> and contributors",
    sitename="MatrixCovers.jl",
    format=Documenter.HTML(;
        canonical="https://HolyLab.github.io/MatrixCovers.jl",
        edit_link="main",
        assets=String[],
    ),
    # `:public` also covers the bindings marked `public` but not exported (the
    # extension hooks). It rests on `Base.ispublic`, which Julia 1.10 lacks, so
    # there the check falls back to the exported set.
    checkdocs=(VERSION >= v"1.11" ? :public : :exports),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/HolyLab/MatrixCovers.jl",
    devbranch="main",
)
