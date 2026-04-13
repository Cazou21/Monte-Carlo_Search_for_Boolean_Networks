using Pkg
using BooleanNetworks

srcdir = abspath(ARGS[1])
modname = ARGS[2]
bnet_files = ARGS[3:length(ARGS)]

moddir = joinpath(srcdir, modname)
modfile = joinpath(moddir, "src", "$(modname).jl")

wd = pwd()

if !isfile(modfile)
cd(srcdir)
Pkg.generate(modname)
cd(wd)
end

io = open(modfile, "w")
BooleanNetworks.make_jl_module(io, modname, bnet_files)

Pkg.develop(path=joinpath(srcdir, modname))
Pkg.activate(moddir)
Pkg.develop(path=dirname(dirname(pathof(BooleanNetworks))))

#Pkg.gc()
#Pkg.activate(wd)
#Pkg.status()

