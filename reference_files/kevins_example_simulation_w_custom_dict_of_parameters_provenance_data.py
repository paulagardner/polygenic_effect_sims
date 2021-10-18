import json
import tskit
import fwdpy11

L = 10000
SEED = 42
params = {
    "recregions": [fwdpy11.PoissonInterval(0, L, 0.25)],
    "sregions": [fwdpy11.GaussianS(0, L, 1.0, sd=0.25)],
    "gvalue": fwdpy11.Additive(
        2, fwdpy11.GSS(optimum=1.0, VS=53.0), fwdpy11.GaussianNoise(mean=-0.1, sd=5)
    ),
    "rates": (0, 1.0, None),
    "simlen": 100,
    "prune_selected": False,
}


params = fwdpy11.ModelParams(**params)
pop = fwdpy11.DiploidPopulation(1000, L)
rng = fwdpy11.GSLrng(SEED)
fwdpy11.evolvets(rng, pop, params, simplification_interval=100)
ts = pop.dump_tables_to_tskit(
    # The actual model params
    model_params=params,
    # Any dict you want.  Some of what I'm putting here is redundant...
    # This dict will get written to the "provenance" table
    parameters={"seed": SEED, "simplification_interval": 100, "meanE": -0.1},
    wrapped=True,
)


# Check the medatdata
# Get the params from the model params
print(ts.model_params.gvalue.gvalue_to_fitness.optimum)
print(ts.model_params.gvalue.gvalue_to_fitness.VS)
# Extract your custom dict out from the provenance table
provenance = json.loads(ts.ts.provenance(0).record)
print("The provenance data are")
print(provenance)
print("The custom parameters dict is in the provenance data, and is also a dict:")
print(provenance["parameters"]["meanE"])