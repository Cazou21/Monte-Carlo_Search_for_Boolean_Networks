module BooleanNetworks
export zerocfg,
        load_bnet,
        load_bnet_ensemble,
        make_mutant,
        fasync_simulations,
        fasync_ping,
        outputs_ratios,
        resolve,
        BooleanNetwork

using Random

const LocalFunctions = Vector{Function}

struct MutatedLocalFunctions
    f::LocalFunctions
    mutated::Vector{Bool} # dim = length(bn.f)
    mutant::Vector{Bool} # dim = length(bnf)
    mutated_idx::BitSet
end

struct BooleanNetwork
    nodes::Vector{String}
    index::Dict{String,Int64}
    f::Union{LocalFunctions,MutatedLocalFunctions}
    n::Int64
    out_influences::Vector{BitSet}
end

resolve(bn, indexes) = [bn.nodes[i] for i in indexes]

zerocfg(bn) = zeros(Bool, length(bn.nodes))

filtered_subvector(v, to_remove) = [setdiff(sv, to_remove) for sv in v]

function make_mutant(bn::BooleanNetwork, d_mutant)
    mutated = zerocfg(bn)
    mutant = zerocfg(bn)
    mutated_idx = BitSet()
    for (a, v) in pairs(d_mutant)
        i = bn.index[a]
        push!(mutated_idx, i)
        mutated[i] = true
        mutant[i] = v
    end
    out_influences = filtered_subvector(bn.out_influences, mutated_idx)
    BooleanNetwork(bn.nodes, bn.index, MutatedLocalFunctions(bn.f, mutated, mutant, mutated_idx), bn.n, out_influences)
end

function make_mutant(bns::AbstractVector{BooleanNetwork}, d_mutant)
    bn1 = make_mutant(bns[1], d_mutant)
    [BooleanNetwork(bn.nodes, bn.index, MutatedLocalFunctions(bn.f, bn1.f.mutated, bn1.f.mutant, bn1.f.mutated_idx), bn.n,
                filtered_subvector(bn.out_influences, bn1.f.mutated_idx))
        for bn in bns]
end

include("io.jl")
include("fasync.jl")
include("utils.jl")

end
