from __future__ import annotations

from typing import Any, Optional

import concurrent.futures
import math
import threading

import grpc
import numpy as np
import volpe_py as _volpe
from deap import base, creator, tools
from opfunu.cec_based.cec2022 import F122022

volpe = _volpe


NDIM = 20
BASE_POPULATION_SIZE = 100

# PSO hyperparameters
INERTIA = 1.0
COGNITIVE = 2.0
SOCIAL = 2.0

func = F122022(ndim=NDIM)
LOW = float(func.lb[0])
HIGH = float(func.ub[0])
VMAX = (HIGH - LOW) * 0.2


if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

if not hasattr(creator, "Particle"):
    creator.create(
        "Particle",
        np.ndarray,
        fitness=creator.FitnessMin,
        smin=None,
        smax=None,
    )

creator_any = creator


def generate_particle(size: int, pmin: float, pmax: float, smin: float, smax: float):
    part = creator_any.Particle(np.random.uniform(pmin, pmax, size).astype(np.float32))
    part.speed = np.random.uniform(smin, smax, size).astype(np.float32)
    part.smin = smin
    part.smax = smax
    part.best = None
    return part


def fitness_value(x: np.ndarray) -> float:
    if np.any(x < LOW) or np.any(x > HIGH):
        return float(np.inf)
    return float(np.float32(func.evaluate(x)))


def evaluate_particle(part) -> tuple[float]:
    return (fitness_value(np.asarray(part, dtype=np.float32)),)


def update_particle(part, best, phi1: float, phi2: float, inertia: float):
    if best is None:
        return
    if part.best is None:
        part.best = clone_particle(part)

    u1 = np.random.uniform(0.0, phi1, len(part)).astype(np.float32)
    u2 = np.random.uniform(0.0, phi2, len(part)).astype(np.float32)
    v_u1 = u1 * (part.best - part)
    v_u2 = u2 * (best - part)

    part.speed = (inertia * part.speed) + v_u1 + v_u2
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)

    part[:] = np.clip(part + part.speed, LOW, HIGH)


toolbox = base.Toolbox()
toolbox.register("particle", generate_particle, size=NDIM, pmin=LOW, pmax=HIGH, smin=-VMAX, smax=VMAX)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("evaluate", evaluate_particle)
toolbox.register("update", update_particle, phi1=COGNITIVE, phi2=SOCIAL, inertia=INERTIA)


def particle_fitness_value(part) -> float:
    if getattr(part.fitness, "values", ()):
        return float(part.fitness.values[0])
    return float(fitness_value(np.asarray(part, dtype=np.float32)))


def clone_particle(part):
    new_part = creator_any.Particle(np.array(part, dtype=np.float32, copy=True))
    new_part.smin = getattr(part, "smin", -VMAX)
    new_part.smax = getattr(part, "smax", VMAX)
    speed = getattr(part, "speed", np.zeros(NDIM, dtype=np.float32))
    new_part.speed = np.array(speed, dtype=np.float32, copy=True)
    if getattr(part.fitness, "values", ()):
        new_part.fitness.values = (float(part.fitness.values[0]),)
    new_part.best = None
    return new_part


def encode_particle(particle) -> bytes:
    best_position = np.asarray(particle.best if particle.best is not None else particle, dtype=np.float32)
    best_fit = (
        float(particle.best.fitness.values[0])
        if particle.best is not None and getattr(particle.best.fitness, "values", ())
        else float(np.inf)
    )
    payload = np.concatenate(
        [
            np.asarray(particle, dtype=np.float32),
            np.asarray(particle.speed, dtype=np.float32),
            best_position,
            np.array([best_fit], dtype=np.float32),
        ]
    )
    return payload.tobytes()


def decode_particle(memb: Any):
    raw = np.frombuffer(memb.genotype, dtype=np.float32)
    full_size = (3 * NDIM) + 1

    if raw.size >= full_size:
        position = raw[:NDIM].copy()
        velocity = raw[NDIM : (2 * NDIM)].copy()
        best_position = raw[(2 * NDIM) : (3 * NDIM)].copy()
        best_fitness = float(raw[(3 * NDIM)])
    else:
        position = raw.copy()
        if position.size != NDIM:
            position = np.resize(position, NDIM).astype(np.float32)
        velocity = np.random.uniform(-VMAX, VMAX, size=NDIM).astype(np.float32)
        best_position = position.copy()
        best_fitness = float(np.inf)

    velocity = np.clip(velocity, -VMAX, VMAX).astype(np.float32)
    best_position = np.clip(best_position, LOW, HIGH).astype(np.float32)
    position = np.clip(position, LOW, HIGH).astype(np.float32)

    fit = float(memb.fitness)
    if not np.isfinite(fit):
        fit = fitness_value(position)

    if not np.isfinite(best_fitness):
        best_fitness = fit

    if fit < best_fitness:
        best_fitness = fit
        best_position = position.copy()

    part = creator_any.Particle(position.astype(np.float32, copy=False))
    part.speed = velocity.astype(np.float32, copy=False)
    part.smin = -VMAX
    part.smax = VMAX
    part.fitness.values = (fit,)

    best_part = creator_any.Particle(best_position.astype(np.float32, copy=False))
    best_part.speed = np.zeros(NDIM, dtype=np.float32)
    best_part.smin = -VMAX
    best_part.smax = VMAX
    best_part.fitness.values = (best_fitness,)
    best_part.best = None

    part.best = best_part
    return part


def pop_list_to_result(popln: list[Any]):
    members = [
        volpe.ResultIndividual(
            representation=np.array2string(np.asarray(mem), precision=6),
            fitness=particle_fitness_value(mem),
        )
        for mem in popln
    ]
    return volpe.ResultPopulation(members=members)


def pop_list_to_bytes(popln: list[Any]):
    members = [
        volpe.Individual(
            genotype=encode_particle(mem),
            fitness=particle_fitness_value(mem),
        )
        for mem in popln
    ]
    return volpe.Population(members=members, problemID="p1")


class VolpeGreeterServicer(volpe.VolpeContainerServicer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.poplock = threading.Lock()
        self.popln: list[Any] = []
        self.global_best: Optional[Any] = None
        self._reset_population(BASE_POPULATION_SIZE)

    def _reset_population(self, size: int):
        self.popln = toolbox.population(n=size)
        self._evaluate_and_refresh_bests()

    def _evaluate_and_refresh_bests(self):
        self.global_best = None
        for part in self.popln:
            part.fitness.values = toolbox.evaluate(part)

            if part.best is None or part.best.fitness.values[0] > part.fitness.values[0]:
                part.best = clone_particle(part)

            if self.global_best is None or self.global_best.fitness.values[0] > part.fitness.values[0]:
                self.global_best = clone_particle(part)

        self._refresh_global_best()

    def _refresh_global_best(self):
        if not self.popln:
            self.global_best = None
            return
        if self.global_best is not None:
            return
        best = min(self.popln, key=particle_fitness_value)
        self.global_best = clone_particle(best)

    def _step_particle(self, particle):
        if self.global_best is None:
            return
        toolbox.update(particle, self.global_best)

        particle.fitness.values = toolbox.evaluate(particle)

        if particle.best is None or particle.best.fitness.values[0] > particle.fitness.values[0]:
            particle.best = clone_particle(particle)

        if self.global_best is None or self.global_best.fitness.values[0] > particle.fitness.values[0]:
            self.global_best = clone_particle(particle)

    def SayHello(self, request: Any, context: grpc.ServicerContext):
        return volpe.HelloReply(message="hello " + request.name)

    def InitFromSeed(self, request: Any, context: grpc.ServicerContext):
        with self.poplock:
            np.random.seed(request.seed)
            self._reset_population(BASE_POPULATION_SIZE)
            return volpe.Reply(success=True)

    def InitFromSeedPopulation(self, request: Any, context: grpc.ServicerContext):
        with self.poplock:
            original_len = len(self.popln) if self.popln is not None else BASE_POPULATION_SIZE
            incoming = [decode_particle(memb) for memb in request.members]

            if self.popln is None:
                self.popln = []
            self.popln.extend(incoming)

            # Keep best individuals by current fitness.
            self.popln.sort(key=particle_fitness_value)
            self.popln = self.popln[:original_len]
            self.global_best = None
            self._refresh_global_best()
            return volpe.Reply(success=True)

    def GetBestPopulation(self, request: Any, context):
        with self.poplock:
            if self.popln is None:
                return volpe.Population(members=[], problemID="p1")
            pop_sorted = sorted(self.popln, key=particle_fitness_value)
            return pop_list_to_bytes(pop_sorted[: request.size])

    def GetResults(self, request: Any, context):
        with self.poplock:
            if self.popln is None:
                return volpe.ResultPopulation(members=[])
            pop_sorted = sorted(self.popln, key=particle_fitness_value)
            return pop_list_to_result(pop_sorted[: request.size])

    def GetRandom(self, request: Any, context):
        with self.poplock:
            if self.popln is None:
                return volpe.Population(members=[], problemID="p1")
            idx = np.random.randint(0, len(self.popln), size=request.size)
            sample = [self.popln[i] for i in idx]
            return pop_list_to_bytes(sample)

    def AdjustPopulationSize(self, request: Any, context: grpc.ServicerContext):
        with self.poplock:
            target = max(0, int(request.size))
            current = len(self.popln)

            if target > current:
                self.popln.extend(toolbox.population(n=target - current))
                self._evaluate_and_refresh_bests()
            elif target < current:
                self.popln.sort(key=particle_fitness_value)
                self.popln = self.popln[:target]
                self.global_best = None

            self._refresh_global_best()
            return volpe.Reply(success=True)

    def RunForGenerations(self, request: Any, context):
        with self.poplock:
            generations = max(0, int(request.size))
            if not self.popln:
                return volpe.Reply(success=True)

            # Classic PSO loop.
            for _ in range(generations):
                for particle in self.popln:
                    self._step_particle(particle)

        return volpe.Reply(success=True)


if __name__ == '__main__':
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
    volpe.add_VolpeContainerServicer_to_server(VolpeGreeterServicer(), server)
    server.add_insecure_port("0.0.0.0:8081")
    server.start()
    server.wait_for_termination()
