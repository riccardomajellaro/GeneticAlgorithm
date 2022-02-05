import sys
import numpy as np
import random

class GeneticAlgorithm():
    def __init__(self, func, budget=None, initial_pop=10, plus=True):
        self.func = func
        if budget is None:
            self.tot_budget = int(50*func.meta_data.n_variables**2)
        else: self.tot_budget = budget
        self.initial_pop = initial_pop
        self.plus = plus

    def init_population(self):
        self.x = np.random.randint(0, 2, size=(self.initial_pop, self.func.meta_data.n_variables))

    def evaluate(self):
        # evaluate solutions and check for run and global optimum
        # then return evaluations (or None if optimum is found)
        f = np.array([self.func(x_i) for x_i in self.x])
        self.budget -= self.x.shape[0]
        for i, f_i in enumerate(f):
            if f_i > self.f_opt_run:
                self.f_opt_run = f_i
                self.x_opt_run = self.x[i]
                print("Score:", round(self.f_opt_run, 3), "at evaluation", self.tot_budget-self.budget+1)
                if self.f_opt_run > self.f_opt:
                    self.f_opt = self.f_opt_run
                    self.x_opt = self.x_opt_run
                if self.f_opt_run >= self.optimum:
                    print(f"{self.tot_budget-self.budget}/{self.tot_budget} evaluation")
                    return None
        return f

    def tournament_selection(self, num_tournaments=10, k=10, p=0.6):
        f = self.evaluate()
        if f is None: return None # optimum found
        x_new = []
        for i in range(num_tournaments):
            indices = random.sample(range(self.x.shape[0]), k)
            x_t, f_t = self.x[indices], f[indices]
            prob = np.random.uniform(0, 1)
            for j in range(k):
                if prob < sum([p*(1-p)**(k) for k in range(j+1)]):
                    x_new.append(x_t[np.argsort(f_t)[-j-1]])
                    break
                if j == k-1: # no one has been selected
                    x_new.append(x_t[np.argmax(f_t)]) # get the fittest
        self.x = np.array(x_new)
        return True

    def proportionate_selection(self, spins=10):
        f = self.evaluate()
        if f is None: return None
        if f.sum() == np.min(f)*f.shape[0] or np.sum(f == f[0]) == f.shape[0]:
            probs = f / f.sum()
        else:
            probs = (f-np.min(f)) / (f.sum() - np.min(f)*f.shape[0])
        x_new = []
        for _ in range(spins):
            roulette = np.random.uniform(0, 1)
            for i in range(probs.shape[0]):
                if roulette < np.sum(probs[:i+1]): 
                    x_new.append(list(self.x[i]))
                    break
        self.x = np.array(x_new)
        return True

    def get_pairs(self, x):
        # permute the indexes of x and get the pairs as
        # evens -> x1   odds -> x2
        indexes = np.random.permutation(self.x.shape[0])
        return [(self.x[indexes[i]], self.x[indexes[i+1]]) for i in range(indexes.size) if i%2 == 0]

    def uniform_crossover(self):
        x_new = self.x.copy()
        for x_1, x_2 in self.get_pairs(x_new):
            for i in range(x_new.shape[1]):
                if np.random.uniform(0, 1) < 0.5:
                    x_1i_tmp = x_1[i]
                    x_1[i] = x_2[i]
                    x_2[i] = x_1i_tmp
        self.x = x_new

    def n_points_crossover(self, n_points=2):
        x_new = self.x.copy()
        for x_1, x_2 in self.get_pairs(x_new):
            # generate the n splits with no repetitions
            split_ind = set(np.random.randint(1, self.x.shape[1], size=n_points))
            while len(split_ind) < n_points:
                split_ind.update(np.random.randint(1, self.x.shape[1]-1, size=n_points-len(split_ind)))
            split_ind = np.sort(list(split_ind))
            for i in range(split_ind.shape[0]):
                if i % 2: continue
                ind = split_ind[i]
                if i == split_ind.shape[0] - 1: right = self.x.shape[1]
                else: right = split_ind[i+1]
                x_1_tmp = x_1.copy()
                x_1[ind:right] = x_2[ind:right]
                x_2[ind:right] = x_1_tmp[ind:right]
        self.x = x_new

    def mutate(self, p_m=None):
        if p_m is None:
            p_m = []
            for x_i in self.x:
                f_x_i = self.func(x_i)
                if 2*(f_x_i+1)-self.x.shape[1]:
                    p_m.append(abs(1 / (2*(f_x_i+1)-self.x.shape[1])))
                else: p_m.append(1 / self.x.shape[1])
            self.budget -= self.x.shape[0]
        else: p_m = [p_m] * self.x.shape[0]
        x_new = np.array([[int(not x_j) if np.random.uniform(0, 1) < p_m[i] else x_j for x_j in x_i] for i, x_i in enumerate(self.x)])

        if self.plus: # plus mode
            self.x = np.r_[self.parents, x_new]
        else: # comma mode
            self.x = x_new

    def __call__(self, mating="proportionate", mating_params=[10], crossover="n-points", cross_params=[2], p_m=None, seed=42):
        if mating == "tournament":
            mating = self.tournament_selection
        else: mating = self.proportionate_selection

        if crossover == "n-points":
            crossover = self.n_points_crossover
        else: crossover = self.uniform_crossover

        if self.func.meta_data.problem_id == 18 and self.func.meta_data.n_variables == 32:
            self.optimum = 8
        else:
            self.optimum = self.func.objective.y
        print(f"Optimum value: {self.optimum}\n")

        # reset optimal solution
        self.f_opt = sys.float_info.min
        self.x_opt = None
        # 10 independent runs for each algorithm on each problem.
        used_budgets = np.array([])
        for r in range(10):
            print("Run", r+1)
            # set seed for reproducibility
            np.random.seed(seed+r)
            random.seed(seed+r)
            # fill budget
            self.budget = self.tot_budget
            # initialize population
            self.init_population()
            # reset otimum of the run and evaluate initialized population
            self.f_opt_run = sys.float_info.min
            if self.evaluate() is None:
                self.func.reset()
                used_budgets = np.append(used_budgets, self.tot_budget-self.budget)
                print(f"_________________\n")
                continue
            # run genetic algorithm until budget is over
            while self.budget > 0:
                if mating(*mating_params) is None: # optimum found
                    break
                self.parents = self.x
                crossover(*cross_params)
                self.mutate(p_m=p_m)
            self.func.reset()
            used_budgets = np.append(used_budgets, self.tot_budget-self.budget)
            print(f"_________________\n")

        print("Max target:", self.f_opt, "with candidate", self.x_opt)
        print("Mean used budget:", used_budgets.mean(), "- Std used budget:", used_budgets.std())