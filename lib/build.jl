# Build the Python and MATLAB bindings from the repository root with Julia 1.13:
#
#   julia --project=lib/build-env lib/build.jl
#
# Instantiate the build environment first:
#
#   julia --project=lib/build-env -e 'using Pkg; Pkg.instantiate()'
#
# The output goes to `lib/out/`: the Python package `matrixcovers/`, the
# MATLAB package `+matrixcovers/` with its MEX gateway source, the C header,
# and the bundled library.

using TOML: TOML

const HERE = @__DIR__
const REPO_ROOT = abspath(joinpath(HERE, ".."))

# juliac copies the project to a temporary directory, so the path to the
# MatrixCovers source must be absolute.
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
    matlab_package = "matrixcovers",
    project = prepare_project(),
    version = REPO_VERSION,
    coerce_arrays = true,
    bundle = get(ENV, "MATRIXCOVERS_BUNDLE", "true") == "true",
    verbose = true,
)

# The Python package's public API is the hand-written `_facade.py`, which
# builds on the generated `_generated.py`.
cp(joinpath(HERE, "python", "_facade.py"),
   joinpath(HERE, "out", "matrixcovers", "_facade.py");
   force = true)

@info "Built matrixcovers" library=result.library bundle=result.bundle_dir
