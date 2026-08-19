# Build the Python wheel from the repository root with Julia 1.13:
#
#   julia +rc --project=lib/build-env lib/build.jl
#
# Instantiate the build environment first:
#
#   julia +rc --project=lib/build-env -e 'using Pkg; Pkg.instantiate()'
#
# The temporary project supplies the absolute source path required by juliac.

using TOML: TOML

const HERE = @__DIR__
const REPO_ROOT = abspath(joinpath(HERE, ".."))

function prepare_project()
    toml = TOML.parsefile(joinpath(HERE, "Project.toml"))
    sources = get(toml, "sources", Dict{String, Any}())
    sources["MatrixCovers"] = Dict("path" => REPO_ROOT)
    toml["sources"] = sources
    tmp = mktempdir(; prefix = "matrixcovers-lib-project-")
    open(joinpath(tmp, "Project.toml"), "w") do io
        TOML.print(io, toml; sorted = true)
    end
    return tmp
end

const REPO_VERSION = TOML.parsefile(joinpath(REPO_ROOT, "Project.toml"))["version"]

using JuliaLibWrapping, JuliaC

result = standard_build(HERE;
    libname = "matrixcovers",
    python_package = "matrixcovers",
    project = prepare_project(),
    version = REPO_VERSION,
    verbose = true,
)

# Replace the generated facade with the public API.
cp(joinpath(HERE, "python", "_facade.py"),
   joinpath(HERE, "out", "matrixcovers", "_facade.py");
   force = true)

@info "Built matrixcovers" library=result.library bundle=result.bundle_dir
