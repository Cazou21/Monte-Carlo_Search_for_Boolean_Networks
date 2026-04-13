import os
from glob import glob

import tomli as tomllib
import jl_impl.jlbns

bnlib_dir = os.path.join(os.path.dirname(__file__), "jlbns")
pkgtarget = os.path.dirname(__file__)

import juliacall
import juliapkg

class BooleanNetworkEnsemble(object):
    def __init__(self, bndir, name, force_rebuild=False):
        self.bndir = bndir
        self.name = name
        self.jlmoddir = os.path.join(bnlib_dir, name)
        self.jlprojfile = os.path.join(self.jlmoddir, "Project.toml")
        self.bnmodfile = os.path.join(self.jlmoddir, "src", f"{name}.jl")

        jl = juliacall.newmodule("setup")
        juliapkg.add("BooleanNetworks", uuid="d3336e3b-cb21-4231-9499-9b18909c4c76",
                     path=os.path.join(os.path.dirname(__file__),"deps", "BooleanNetworks.jl"),
                     dev=True, target=pkgtarget)
        juliapkg.resolve()

        if not os.path.isfile(self.bnmodfile) or force_rebuild:
            self.generate_jl()

        with open(self.jlprojfile, "rb") as fp:
            pkgspec = tomllib.load(fp)
        juliapkg.add(self.name, uuid=pkgspec['uuid'], dev=True,
                     path=os.path.join(bnlib_dir, self.name),
                     target=pkgtarget)
        juliapkg.resolve()

        self.jl = juliacall.newmodule("session")
        self.jl.seval(f"using {name}")
        self.jlmod = getattr(self.jl, name)
        self.size = self.jl.length(self.jlmod.ensemble)

    def generate_jl(self):
        jl = juliacall.newmodule("setup")
        if not os.path.isfile(self.bnmodfile):
            wd = jl.pwd()
            jl.cd(bnlib_dir)
            jl.seval("using Pkg")
            jl.Pkg.generate(self.name) #(f"Pkg.generate(\"{self.name}\")")
            jl.cd(wd)

            with open(self.jlprojfile, "a") as fp:
                fp.write("\n[deps]\nBooleanNetworks = \"d3336e3b-cb21-4231-9499-9b18909c4c76\"\n")
        jl.seval("using BooleanNetworks")
        io = jl.open(self.bnmodfile, "w")
        jl.BooleanNetworks.make_jl_module(io, self.name, glob(f"{self.bndir}/*.bnet"))
        jl.close(io)

class Experiment(object):
    def __init__(self, main_ensemble, size, sim_per, maxsteps, init, target,
                 init_rand=None, sample_at_evaluate=False):
        """
        init: either dict Node => True/False or Bnet_name => dict
        """
        self.jl = main_ensemble.jl
        self.jl.seval(f"using BooleanNetworks")
        self.jl.seval(f"using Random")
        self.size = size
        self._init = init
        self.sim_per = sim_per
        self.maxsteps = maxsteps
        self.main_ensemble = main_ensemble
        self.sample_at_evaluate = sample_at_evaluate

        self.jl.seval("is_homegoenous_ensemble(ens) = allequal([bn.nodes for bn in ens])")
        self.jl.seval(f"@assert is_homegoenous_ensemble({main_ensemble.name}.ensemble)")
        self.jl.seval("encode_cfg(bn, cfg) = [Bool(get(cfg, bn.nodes[i], 0)) for i in 1:bn.n]")
        self.jl.seval("encode_cfgs(bns, names, cfgs) = "\
                        "[encode_cfg(bns[i], cfgs[replace(name, \".bnet\" => \"\")])"\
                        " for (i,name) in enumerate(names)]")

        self._prepare()

        self.jl_fasync_args = {}
        self.target1 = [self.bns[0].index[a] for a, v in target.items() if v]
        self.target0 = [self.bns[0].index[a] for a, v in target.items() if not v]
        if init_rand:
            self.init_rand = {self.bns[0].index[a]: p for a, p in init_rand.items()}
            self.jl_fasync_args["rand_x"] = self.init_rand
        # warmup
        self.jl.fasync_ping(self.bns, 1, 2, self.init, self.target1,
                            self.target0, **self.jl_fasync_args)

    def _prepare(self):
        if self.size < self.main_ensemble.size:
            self.mask = self.jl.seval(f"mask = random_mask({self.main_ensemble.size}, {self.size})")
            self.bns = self.jl.seval(f"@view {self.main_ensemble.name}.ensemble[mask]")
            self.names = self.jl.seval(f"@view {self.main_ensemble.name}.names[mask]")
        else:
            self.bns = self.main_ensemble.jlmod.ensemble
            self.names = self.main_ensemble.jlmod.names
        self.actual_size = len(self.bns)

        if isinstance(next(iter(self._init.values())), dict): # init is per-model
            if not hasattr(self, "init") or self.size < self.main_ensemble.size:
                self.init = self.jl.encode_cfgs(self.bns, self.names, self._init)
                self.init_per = True
        elif not hasattr(self, "init"):
            self.init = self.jl.encode_cfg(self.bns[0], self._init)
            self.init_per = False

    def evaluate(self, mutant):
        if self.sample_at_evaluate:
            self._prepare()
        bns_m = self.jl.make_mutant(self.bns, mutant)
        count = self.jl.fasync_ping(bns_m, self.sim_per, self.maxsteps,
                                    self.init, self.target1, self.target0,
                                    **self.jl_fasync_args)
        return count/(self.sim_per*self.actual_size)

