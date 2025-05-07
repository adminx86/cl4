
import numpy as np, random
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = (np.random.uniform(150, 200, (100, 3)) - 175) / 25
y = 0.3*X[:,0] - 0.2*X[:,1] + 0.1*X[:,2] + np.random.normal(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def create_nn(params):
    m = MLPRegressor(hidden_layer_sizes=(5,), max_iter=100, solver='lbfgs', warm_start=True)

    m.fit(X_train, y_train)
    p = np.array(params); i = 0
    for arr in m.coefs_ + m.intercepts_:
        s = np.prod(arr.shape)
        arr[...] = p[i:i+s].reshape(arr.shape)
        i += s
    return m

def fitness(p): 
    return mean_squared_error(y_train, create_nn(p).predict(X_train))
def crossover(p1, p2): 
    pt = random.randint(0, len(p1)-1); return p1[:pt]+p2[pt:]
def mutate(p, r=0.1): 
    return [w+np.random.randn()*r if random.random()<0.1 else w for w in p]

tmp = MLPRegressor(hidden_layer_sizes=(5,), max_iter=100); tmp.fit(X_train, y_train)
n_params = sum(np.prod(a.shape) for a in tmp.coefs_ + tmp.intercepts_)
pop = [np.random.uniform(-1, 1, n_params).tolist() for _ in range(10)]

mse_per_gen = []

for g in range(10):
    pop = sorted(pop, key=fitness)
    best_mse = fitness(pop[0])
    mse_per_gen.append(best_mse)
    print(f"Gen {g+1}, MSE: {best_mse:.4f}")

    new_pop = pop[:2]
    while len(new_pop) < len(pop):
        c = mutate(crossover(*random.sample(pop[:5], 2)))
        new_pop.append(c)
    pop = new_pop

# Final output of all generations
print("\nMSE per Generation:")
for i, mse in enumerate(mse_per_gen, 1):
    print(f"Generation {i}: MSE = {mse:.4f}")
