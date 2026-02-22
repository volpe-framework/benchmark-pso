import main as gt

import numpy as np

serv = gt.VolpeGreeterServicer()

serv.InitFromSeed(gt.volpe.Seed(seed=1), None)
print("initialized")
serv.AdjustPopulationSize(gt.volpe.PopulationSize(size=200), None)
print("adjusted size")
for i in range(10):
    serv.RunForGenerations(None, None)
    print(i)
print("ran for gens")
res = serv.GetBestPopulation(gt.volpe.PopulationSize(size=10), None)
serv.InitFromSeedPopulation(res, None)
res = serv.GetBestPopulation(gt.volpe.PopulationSize(size=10), None)

for mem in res.members:
    print(mem.fitness)

result = serv.GetResults(gt.volpe.PopulationSize(size=10), None)
for mem in result.members:
    print(mem.representation, mem.fitness)
