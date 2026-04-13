
using BooleanNetworks

import Random
Random.seed!(1234)

bnet = "data/TumourInvasion/WT_ensemble/bn10.bnet"
@time bn = load_bnet(bnet)

init_active = ["miR200", "miR203", "miR34"]
x0 = zerocfg(bn)
for a in init_active
    x0[bn.index[a]] = true
end
target0 = [bn.index[x] for x in ["Apoptosis"]]
target1 = [bn.index[x] for x in ["Invasion","EMT"]]

nb_sims = 10_000
#nb_sims = 1
maxsteps = 300

println("Warmup...")
@time fasync_ping(bn, 1, 2, x0, target1, target0)

# wild type
println("~ Ping with $nb_sims simulations")
for _ in 1:5
    @time count = fasync_ping(bn, nb_sims, maxsteps, x0, target1, target0)
    println("- ping = $(100*count / nb_sims)%")
end

#using Profile
#@profile fasync_ping(bn, nb_sims, maxsteps, x0, target1, target0)
#Profile.print()

# mutant
mutant = Dict("NICD" => true, "p53" => false)
bn_m = make_mutant(bn, mutant)
println("~ Ping with $nb_sims simulations of mutant $mutant")
for _ in 1:5
    @time count = fasync_ping(bn_m, nb_sims, maxsteps, x0, target1, target0)
    println("- ping = $(100*count / nb_sims)%")
end

