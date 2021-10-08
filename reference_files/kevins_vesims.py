import fwdpy11
import numpy as np

N = 1000
MU = 1e-3
REFNOISE = 0.05
REFVS = 1

EVG = 4 * MU * REFVS
REFH2 = EVG / (EVG + REFNOISE ** 2)

CONSTANT = ["VG", "H2"]

print("SD CONSTANT VG VE H2")

for noise in [0.05, 0.10]:
    VE = noise ** 2
    i = 0
    for VS in [1.0, REFH2 * VE / (4 * MU * (1.0 - REFH2))]:
        pdict = {
            "sregions": [fwdpy11.GaussianS(0, 1, 1, 0.25)],
            "nregions": [],
            "recregions": [fwdpy11.PoissonInterval(0, 1, 1e-3)],
            "rates": (0, MU, None),
            "gvalue": [
                fwdpy11.Additive(2, fwdpy11.GSS(0.0, VS), fwdpy11.GaussianNoise(noise, 0.0))
            ],
            "prune_selected": False,
            "simlen": 10 * N,
        }

        for rep in range(25):
            pop = fwdpy11.DiploidPopulation(N, 1.0)
            seed = np.random.randint(0, 100000, 1)[0]
            rng = fwdpy11.GSLrng(seed)
            params = fwdpy11.ModelParams(**pdict)
            fwdpy11.evolvets(rng, pop, params, 100)
            md = np.array(pop.diploid_metadata, copy=False)
            h2 = md["g"].var() / ((md["g"] + md["e"]).var())
            print(noise, CONSTANT[i], md["g"].var(), md["e"].var(), h2)

        i += 1
