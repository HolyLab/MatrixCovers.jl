using ScaleInvariantAnalysis
using Documenter
using JuMP, HiGHS
using Unitful   # loaded here so the doctests' own `using` cannot emit precompilation output

DocMeta.setdocmeta!(ScaleInvariantAnalysis, :DocTestSetup, :(using ScaleInvariantAnalysis); recursive=true)

makedocs(;
    modules=[ScaleInvariantAnalysis],
    authors="Tim Holy <tim.holy@gmail.com> and contributors",
    sitename="ScaleInvariantAnalysis.jl",
    format=Documenter.HTML(;
        canonical="https://HolyLab.github.io/ScaleInvariantAnalysis.jl",
        edit_link="main",
        assets=String[],
    ),
    checkdocs=:exports,
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/HolyLab/ScaleInvariantAnalysis.jl",
    devbranch="main",
)
