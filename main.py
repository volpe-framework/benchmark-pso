from typing import Any
from typing_extensions import override

import volpe_py as volpe
import grpc
import concurrent.futures
import threading

from opfunu.cec_based.cec2022 import *
from deap import base, creator, tools, algorithms

NDIM=20

func = F72022(ndim=NDIM)

LOW = func.lb[0]
HIGH = func.ub[0]
MUTATE_STD = 5.0
MUTATION_RATE = 0.5
INDPB = 1/NDIM
CXPROB = 0.5

BASE_POPULATION_SIZE = 50
LAMBDA_SIZE = 7*BASE_POPULATION_SIZE

import numpy as np

# Setup DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, LOW, HIGH)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=NDIM)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def fitness(x):
    if any(x > HIGH) or any(x < LOW):
        return (float(np.inf),)
    return (float(np.float32(func.evaluate(x))),)

toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxUniform, indpb=INDPB)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=MUTATE_STD, indpb=INDPB)
toolbox.register("select", tools.selTournament, tournsize=3)

def gen_ind():
    ind = creator.Individual(np.random.uniform(LOW, HIGH, size=NDIM).astype(np.float32))
    ind.fitness.values = fitness(ind)
    return ind

def expand(popln: list, newPop: int):
    if len(popln) == 0:
        return [ gen_ind() for _ in range(newPop) ]
    if len(popln) >= newPop:
        return popln
    while len(popln) < newPop:
        idx1 = np.random.randint(len(popln))
        idx2 = np.random.randint(len(popln))
        x1 = toolbox.clone(popln[idx1])
        x2 = toolbox.clone(popln[idx2])
        toolbox.mate(x1, x2)
        x1.fitness.values = fitness(x1)
        x2.fitness.values = fitness(x2)
        popln.append(x1)
        popln.append(x2)
    return popln

def popListTostring(popln: list):
    indList : list[volpe.ResultIndividual] = []
    for mem in popln:
        indList.append(
                volpe.ResultIndividual(representation=np.array_str(np.array(mem)), 
                                    fitness=mem.fitness.values[0])
                )
    return volpe.ResultPopulation(members=indList)

def bstringToPopln(popln: volpe.Population):
    popList = []
    for memb in popln.members:
        ind = creator.Individual(np.frombuffer(memb.genotype, dtype=np.float32))
        ind.fitness.values = (memb.fitness,)
        popList.append(ind)
    return popList

def popListToBytes(popln: list):
    indList : list[volpe.Individual] = []
    for mem in popln:
        indList.append(volpe.Individual(genotype=np.array(mem).astype(np.float32).tobytes(), fitness=mem.fitness.values[0]))
    return volpe.Population(members=indList, problemID="p1")

class VolpeGreeterServicer(volpe.VolpeContainerServicer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.popln = [ gen_ind() for _ in range(BASE_POPULATION_SIZE)  ]
        self.poplock = threading.Lock()

    @override
    def SayHello(self, request: volpe.HelloRequest, context: grpc.ServicerContext):
        return volpe.HelloReply(message="hello " + request.name)
    @override
    def InitFromSeed(self, request: volpe.Seed, context: grpc.ServicerContext):
        """Missing associated documentation comment in .proto file."""
        with self.poplock:
            self.popln = [ gen_ind() for _ in range(BASE_POPULATION_SIZE)  ]
            return volpe.Reply(success=True)
    @override
    def InitFromSeedPopulation(self, request: volpe.Population, context: grpc.ServicerContext):
        """Missing associated documentation comment in .proto file."""
        with self.poplock:
            ogLen = len(self.popln)
            seedPop = bstringToPopln(request)
            if self.popln == None:
                self.popln = []
            self.popln.extend(seedPop)

            self.popln = toolbox.select(self.popln, ogLen)

            return volpe.Reply(success=True)
    @override
    def GetBestPopulation(self, request: volpe.PopulationSize, context):
        """Missing associated documentation comment in .proto file."""
        with self.poplock:
            if self.popln is None:
                return volpe.Population(members=[], problemID="p1")
            popSorted = sorted(self.popln, key=lambda x: x.fitness.values[0])
            return popListToBytes(popSorted[:request.size])
    @override
    def GetResults(self, request: volpe.PopulationSize, context):
        with self.poplock:
            if self.popln is None:
                return volpe.Population(members=[], problemID="p1")
            popSorted = sorted(self.popln, key=lambda x: x.fitness.values[0])
            return popListTostring(popSorted[:request.size])
    @override
    def GetRandom(self, request: volpe.PopulationSize, context):
        with self.poplock:
            if self.popln is None:
                return volpe.Population(members=[], problemID="p1")
            popList = [self.popln[np.random.randint(len(self.popln))] for _ in range(request.size)]
            return popListToBytes(popList)
    @override
    def AdjustPopulationSize(self, request: volpe.PopulationSize, context: grpc.ServicerContext):
        """Missing associated documentation comment in .proto file."""
        pass
    @override
    def RunForGenerations(self, request: volpe.PopulationSize, context):
        """Missing associated documentation comment in .proto file."""
        with self.poplock:
            # Use DEAP's eaMuPlusLambda algorithm
            self.popln, _ = algorithms.eaMuPlusLambda(
                self.popln, 
                toolbox, 
                mu=len(self.popln),
                lambda_=LAMBDA_SIZE,
                cxpb=CXPROB,
                mutpb=MUTATION_RATE,
                ngen=request.size,
                stats=None,
                halloffame=None,
                verbose=False
            )
                
        return volpe.Reply(success=True)

if __name__=='__main__':
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
    volpe.add_VolpeContainerServicer_to_server(VolpeGreeterServicer(), server)
    server.add_insecure_port("0.0.0.0:8081")
    server.start()
    server.wait_for_termination()
