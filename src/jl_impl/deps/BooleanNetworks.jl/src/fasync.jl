

mutable struct FAsyncSimulationContext
    const x::Vector{Bool}
    tbc::BitSet
    const c::BitSet
end
new_fasync_simulation(n) = FAsyncSimulationContext(Vector{Bool}(undef, n), BitSet(), BitSet())

function _fasync_choose_step(ctx, bn)
    if isempty(ctx.c)
        0
    else
        if length(ctx.c) == 1
            i = pop!(ctx.c)
        else
            i = rand(ctx.c)
            delete!(ctx.c, i)
        end
        ctx.x[i] = !ctx.x[i]
        ctx.tbc = bn.out_influences[i]
        setdiff!(ctx.c, ctx.tbc)
        i
    end
end

function _fasync_step!(ctx, bn, f::LocalFunctions)
    for i in ctx.tbc
        if f[i](ctx.x) ⊻ ctx.x[i]
            push!(ctx.c, i)
        end
    end
    _fasync_choose_step(ctx, bn)
end

function _fasync_step!(ctx, bn, f::MutatedLocalFunctions)
    for i in ctx.tbc
        if f.mutated[i]
            if f.mutant[i] ⊻ ctx.x[i]
                push!(ctx.c, i)
            end
        elseif f.f[i](ctx.x) ⊻ ctx.x[i]
            push!(ctx.c, i)
        end
    end
    _fasync_choose_step(ctx, bn)
end

function _fasync_unroll!(ctx, bn, maxsteps)
    empty!(ctx.c)
    for _ in 1:maxsteps
        if _fasync_step!(ctx, bn, bn.f) == 0
            break
        end
    end
end

function fasync_ping(bn::BooleanNetwork, nb_sims, maxsteps, x, cond1, cond0; rand_x = nothing, ctx = nothing, allidx = nothing)
    s = 0
    if ctx === nothing
        ctx = new_fasync_simulation(bn.n)
    end
    if allidx === nothing
        allidx = BitSet(1:bn.n)
    end
    for i in 1:nb_sims
        ctx.tbc = allidx
        ctx.x[1:bn.n] = x
        if rand_x !== nothing
            for (i,p) in rand_x
                ctx.x[i] = rand() <= p
            end
        end
        _fasync_unroll!(ctx, bn, maxsteps)
        if check_cond(cond1, cond0, ctx.x)
            s += 1
        end
    end
    s
end

function fasync_ping(bns::AbstractVector{BooleanNetwork}, nb_sims_per_model, maxsteps, x::Vector{Bool}, cond1, cond0; rand_x = nothing)
    s = 0
    ctx = new_fasync_simulation(bns[1].n)
    allidx = BitSet(1:bns[1].n)
    for bn in bns
        s += fasync_ping(bn, nb_sims_per_model, maxsteps, x, cond1, cond0; rand_x, ctx, allidx)
    end
    s
end

function fasync_ping(bns::AbstractVector{BooleanNetwork}, nb_sims_per_model, maxsteps, xs::AbstractVector{Vector{Bool}}, cond1, cond0; rand_x=nothing)
    s = 0
    ctx = new_fasync_simulation(bns[1].n)
    allidx = BitSet(1:bns[1].n)
    for i in 1:length(bns)
        s += fasync_ping(bns[i], nb_sims_per_model, maxsteps, xs[i], cond1, cond0; rand_x, ctx, allidx)
    end
    s
end

function check_cond(cond1, cond0, x)
    for i in cond1
        if !x[i]
            return false
        end
    end
    for i in cond0
        if x[i]
            return false
        end
    end
    return true
end

###
### For computing reachable attractors projected on outputs
###

function _fasync_simulation!(ctx, f, n, x, maxsteps, outputs)
    ctx.x[1:n] = x
    _fasync_unroll!(ctx, f, n, maxsteps)
    @view ctx.x[outputs]
end

function pack_output(binarray)
    x::Int64 = 0
    for i in 1:length(binarray)
        x |= binarray[i] << (i-1)
    end
    x
end
unpack_output(x, letters) = [a for (i,a) in enumerate(letters) if x & (1 << (i-1)) > 0]

function fasync_simulations(bn, outputs, nb_sims, maxsteps, x)
    f = bn.f
    n = bn.n
    res = Array{Int64}(undef, nb_sims)
    ctx = new_fasync_simulation(n) #[new_fasync_simulation(n) for _ in 1:Threads.maxthreadid()]
    #Threads.@threads :static for i in 1:nb_sims
    for i in 1:nb_sims
        res[i] = pack_output(_fasync_simulation!(ctx, f, n, x, maxsteps, outputs))
        #res[i] = pack_output(_fasync_simulation!(ctx[Threads.threadid()], f, n, x, maxsteps, outputs))
    end
    res
end

function outputs_ratios(result, outputs, bn)
    cv = zeros(Int64, 2^length(outputs))
    for k in result
        cv[k] += 1
    end
    make_key(k) = resolve(bn, unpack_output(k, outputs))
    Dict(make_key(k) => c/length(result) for (k,c) in enumerate(cv) if c > 0)
end
