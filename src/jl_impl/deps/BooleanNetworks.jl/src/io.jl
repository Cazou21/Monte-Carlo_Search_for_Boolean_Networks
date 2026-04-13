BNET_SYMBOL_RE = r"[A-Za-z][\w\.:]+"

impl_expr_bin(index, expr) = "x -> " * replace(expr,
        "!" => "~",
        r"\b0\b" => "false",
        r"\b1\b" => "true",
         BNET_SYMBOL_RE => x -> "x[$(index[x])]")
impl_expr_bool(index, expr) = "x -> " * replace(expr,
        "|" => "||",
        "&" => "&&",
        r"\b0\b" => "false",
        r"\b1\b" => "true",
         BNET_SYMBOL_RE => x -> "x[$(index[x])]")


function _parse_bnet(filename; sep=",", impl_binary=true)
    data = [split(line, sep) for line in eachline(filename)]
    nodes = [strip(d[1]) for d in data]
    index = Dict(((n,i) for (i,n) in enumerate(nodes)))
    data, nodes, index
end

function make_out_influences(data, index)
    make_influences(expr) = [index[m.match] for m in eachmatch(BNET_SYMBOL_RE, expr)]
    in_influences = [make_influences(d[2]) for d in data]
    n = length(data)
    out_influences = [BitSet() for _ in 1:n]
    for i in 1:n
        for j in in_influences[i]
            push!(out_influences[j], i)
        end
    end
    out_influences
end

function load_bnet(filename; sep=",", impl_binary=true)
    data, nodes, index = _parse_bnet(filename, sep=sep, impl_binary=impl_binary)

    impl_expr = impl_binary ? impl_expr_bin : impl_expr_bool
    make_expr = eval ∘ Meta.parse ∘ impl_expr
    n = length(data)
    f = Vector{Function}(undef, n)
    for i in 1:n
        f[i] = make_expr(index, data[i][2])
    end
    out_influences = make_out_influences(data, index)

    BooleanNetwork(nodes, index, f, n, out_influences)
end

load_bnet_ensemble(filenames, sep=",", impl_binary=true) =
    [load_bnet(filename; sep=sep, impl_binary=impl_binary)
        for filename in filenames]

function _make_jl_bnet!(io, suffix, filename, cache; sep=",", impl_binary=true)
    println(io, "")
    println(io, "# $(filename)")

    data, nodes, index = _parse_bnet(filename)
    out_influences = make_out_influences(data, index)

    println(io, "const bn$(suffix)_name = \"$(basename(filename))\"")
    println(io, "const bn$(suffix)_nodes = $(nodes)")
    println(io, "const bn$(suffix)_index = $(index)")
    impl_expr = impl_binary ? impl_expr_bin : impl_expr_bool
    n = length(data)
    for i in 1:n
        fi = impl_expr(index, data[i][2])
        var = "bn$(suffix)_f_$(i)"
        if haskey(cache, fi)
            println(io, "const $(var) = $(cache[fi])")
        else
            println(io, "const $(var)$(replace(fi, "x ->" => "(x) ="))")
            println(io, "precompile($(var), (Vector{Bool},))")
            cache[fi] = var
        end
    end
    println(io, "const bn$(suffix)_f = [$(join(["bn$(suffix)_f_$(i)" for i in 1:n], ", "))]")
    println(io, "const bn$(suffix)_out_influences = $(out_influences)")
    println(io, "const bn$(suffix) = BooleanNetwork(bn$(suffix)_nodes, bn$(suffix)_index, bn$(suffix)_f, $(n), bn$(suffix)_out_influences)")
    println(io, "")
    "bn$(suffix)"
end

function make_jl_module(io, modname, filenames, sep=",", impl_binary=true)
    println(io, "module $(modname)")
    println(io, "using BooleanNetworks")

    cache = Dict{String,String}()

    bns = String[]
    for (i, filename) in enumerate(filenames)
        var = _make_jl_bnet!(io, "_$(i)", filename, cache, sep=sep, impl_binary=impl_binary)
        push!(bns, var)
    end
    println(io, "const ensemble = [$(join(bns, ", "))]")
    println(io, "const names = [$(join(map(x -> "$(x)_name", bns), ", "))]")
    println(io, "")
    println(io, "end")
end
