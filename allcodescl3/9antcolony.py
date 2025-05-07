import random, math

cities = {0: (0, 0), 1: (1, 5), 2: (5, 2), 3: (6, 6), 4: (8, 3)}

distance = {(i, j): math.dist(cities[i], cities[j]) for i in cities for j in cities if i != j}
pheromone = {edge: 1.0 for edge in distance}


num_ants, num_iters = 10, 100
alpha, beta = 1.0, 5.0
evaporation, Q = 0.5, 100
best_path, best_length = None, float("inf")

def choose_next(curr, visited):
    probs = [(c, (pheromone[(curr, c)] ** alpha) * ((1 / distance[(curr, c)]) ** beta))
             for c in cities if c not in visited]
    total = sum(p for _, p in probs)
    r = random.uniform(0, total)
    s = 0
    for city, prob in probs:
        s += prob
        if r <= s: return city
    return probs[-1][0]  

def build_tour():
    path = [random.choice(list(cities))]
    visited = set(path)
    while len(path) < len(cities):
        next_city = choose_next(path[-1], visited)
        path.append(next_city)
        visited.add(next_city)
    return path + [path[0]]  


for _ in range(num_iters):
    all_tours = []
    for _ in range(num_ants):
        t = build_tour()
        l = sum(distance[(t[i], t[i+1])] for i in range(len(t)-1))
        all_tours.append((t, l))
        if l < best_length: best_path, best_length = t, l
    for edge in pheromone: pheromone[edge] *= (1 - evaporation)
    for path, length in all_tours:
        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            pheromone[(a, b)] += Q / length
            pheromone[(b, a)] += Q / length  

print("\nBest Tour Found:")
print(" -> ".join(map(str, best_path)))
print(f"Total Distance: {best_length:.2f}")
