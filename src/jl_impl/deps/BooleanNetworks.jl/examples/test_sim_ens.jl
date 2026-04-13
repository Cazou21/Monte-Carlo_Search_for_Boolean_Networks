
using BooleanNetworks

import Random
Random.seed!(1234)

ens_dir = "data/TumourInvasion/WT_ensemble"
bnets = readdir(ens_dir, join=true)[1:10]
n = length(bnets)
println("Loading $n bnets...")
@time bns = load_bnet_ensemble(bnets)
@time bns = load_bnet_ensemble(bnets)
println("Done")

init_active = ["miR200", "miR203", "miR34"]
x0 = zerocfg(bns[1])
for a in init_active
    x0[bns[1].index[a]] = true
end
rand_x = Dict(bns[1].index["DNAdamage"] => 0.5, bns[1].index["ECMicroenv"] => 0.5)

target0 = [bns[1].index[x] for x in ["Apoptosis"]]
target1 = [bns[1].index[x] for x in ["CellCycleArrest","Invasion","EMT"]]

nb_sims = 10_000
maxsteps = 300

# warm-up (TODO: should be done at compilation..)
println("Warmup...")
@time fasync_ping(bns, 1, 2, x0, target1, target0)
@time fasync_ping(bns, 1, 2, x0, target1, target0; rand_x = rand_x)
println("Warmup done.")

nb_sims_per = max(1, nb_sims ÷ length(bns))


# wild type
println("~ Ping $(length(bns)) BNs with $nb_sims_per simulation each")
for _ in 1:5
    @time count = fasync_ping(bns, nb_sims_per, maxsteps, x0, target1, target0; rand_x = rand_x)
    println("- ping = $(100*count / (nb_sims_per*length(bns)))%")
end


# mutant
mutant = Dict("NICD" => true, "p53" => false)
bns_m = make_mutant(bns, mutant)
println("~ Ping with $nb_sims simulations of mutant $mutant")
for _ in 1:5
    @time count = fasync_ping(bns_m, nb_sims_per, maxsteps, x0, target1, target0; rand_x = rand_x)
    println("- ping = $(100*count / (nb_sims_per*length(bns)))%")
end

