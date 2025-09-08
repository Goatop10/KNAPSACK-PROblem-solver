import numpy as np, random

def fitness(ind, v, w, cap):
    tw, tv = np.sum(ind * w), np.sum(ind * v)
    return tv if tw <= cap else 0

def select(pop, fits):
    return pop[max(random.sample(range(len(pop)), 3), key=lambda i: fits[i])]

def crossover(p1, p2):
    pt = random.randint(1, len(p1) - 1)
    return np.hstack((p1[:pt], p2[pt:])), np.hstack((p2[:pt], p1[pt:]))

def mutate(ind, rate=0.05):
    for i in range(len(ind)):
        if random.random() < rate: ind[i] ^= 1
    return ind

def knapsack_ga(values, weights, cap, pop_size=50, gens=200, rate=0.05):
    n = len(values)
    pop = [np.random.randint(0, 2, n) for _ in range(pop_size)]

    for _ in range(gens):
        fits = [fitness(ind, values, weights, cap) for ind in pop]
        new_pop = []
        while len(new_pop) < pop_size:
            c1, c2 = crossover(select(pop, fits), select(pop, fits))
            new_pop += [mutate(c1, rate), mutate(c2, rate)]
        pop = new_pop[:pop_size]

    fits = [fitness(ind, values, weights, cap) for ind in pop]
    best = pop[np.argmax(fits)]
    return best, max(fits)

# Example Run
if __name__ == "__main__":
    values = np.array([60, 100, 120])
    weights = np.array([10, 20, 30])
    capacity = 50

    best_solution, best_value = knapsack_ga(values, weights, capacity)
    print("Best Solution (0=exclude,1=include):", best_solution)
    print("Best Value:", best_value)
