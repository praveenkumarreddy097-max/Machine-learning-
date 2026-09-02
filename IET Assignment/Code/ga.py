"""
ga.py
A Genetic Algorithm implemented from scratch (no optimisation libraries)
to search for a good initial weight vector and key hyperparameters
(number of hidden units, learning rate) for the MLP.

Chromosome encoding:
    [ w_1, w_2, ..., w_P, h_code, lr_code ]
    - w_i  : real-valued genes, the flattened initial weights/biases of the
             MLP (length P = n_params for the *maximum* hidden-unit budget)
    - h_code : integer gene mapped to hidden-unit count in {4, 8, 12, 16}
    - lr_code: integer gene mapped to learning rate in
               {0.001, 0.01, 0.05, 0.1}

Fitness function:
    fitness(chromosome) = mean validation accuracy over a short "proxy"
    training run (few epochs, small held-out split) MINUS a small
    complexity penalty on hidden-unit count (Occam's-razor style term)
    to discourage unnecessarily large networks that would just memorise
    the proxy split.

Operators:
    - Selection : tournament selection (size k)
    - Crossover : uniform crossover on real genes, single-point on the
                  discrete hyperparameter genes
    - Mutation  : Gaussian perturbation on real genes, random-reset on
                  discrete genes, both applied at rate p_mut
    - Elitism   : the best individual of each generation is carried over
                  unmodified (prevents fitness from ever decreasing)
"""
import numpy as np
from .mlp import MLP

HIDDEN_CHOICES = [4, 8, 12, 16]
LR_CHOICES = [0.001, 0.01, 0.05, 0.1]
MAX_HIDDEN = max(HIDDEN_CHOICES)


def decode_hidden(code):
    return HIDDEN_CHOICES[int(code) % len(HIDDEN_CHOICES)]


def decode_lr(code):
    return LR_CHOICES[int(code) % len(LR_CHOICES)]


class GeneticWeightSearch:
    def __init__(self, n_input, pop_size=24, generations=25,
                 p_crossover=0.8, p_mut=0.1, tournament_k=3,
                 complexity_penalty=0.002, seed=0):
        self.n_input = n_input
        self.pop_size = pop_size
        self.generations = generations
        self.p_crossover = p_crossover
        self.p_mut = p_mut
        self.tournament_k = tournament_k
        self.complexity_penalty = complexity_penalty
        self.rng = np.random.default_rng(seed)
        self.n_weight_genes = MLP.n_params(n_input, MAX_HIDDEN)
        self.history_best = []
        self.history_mean = []

    def _random_individual(self):
        weight_genes = self.rng.normal(0, 0.5, size=self.n_weight_genes)
        h_gene = self.rng.integers(0, len(HIDDEN_CHOICES))
        lr_gene = self.rng.integers(0, len(LR_CHOICES))
        return np.concatenate([weight_genes, [h_gene, lr_gene]])

    def _build_mlp(self, individual):
        n_hidden = decode_hidden(individual[-2])
        lr = decode_lr(individual[-1])
        weight_genes = individual[:-2]
        # Slice the correctly-sized sub-vector for this hidden-unit count
        n_needed = MLP.n_params(self.n_input, n_hidden)
        sub = weight_genes[:n_needed]
        net = MLP(self.n_input, n_hidden)
        net.set_flat_weights(sub)
        return net, lr, n_hidden

    def fitness(self, individual, X_tr, y_tr, X_val, y_val, proxy_epochs=15):
        net, lr, n_hidden = self._build_mlp(individual)
        net.fit(X_tr, y_tr, epochs=proxy_epochs, lr=lr, batch_size=32, l2=1e-3)
        preds = net.predict(X_val)
        acc = float(np.mean(preds == y_val))
        penalty = self.complexity_penalty * (n_hidden / MAX_HIDDEN)
        return acc - penalty

    def _tournament_select(self, pop, fitnesses):
        idx = self.rng.choice(len(pop), size=self.tournament_k, replace=False)
        best_i = idx[np.argmax(fitnesses[idx])]
        return pop[best_i]

    def _crossover(self, p1, p2):
        if self.rng.random() > self.p_crossover:
            return p1.copy(), p2.copy()
        c1, c2 = p1.copy(), p2.copy()
        # uniform crossover on real weight genes
        mask = self.rng.random(self.n_weight_genes) < 0.5
        c1[:self.n_weight_genes] = np.where(mask, p1[:self.n_weight_genes], p2[:self.n_weight_genes])
        c2[:self.n_weight_genes] = np.where(mask, p2[:self.n_weight_genes], p1[:self.n_weight_genes])
        # single-point crossover on the two discrete genes
        if self.rng.random() < 0.5:
            c1[-2], c2[-2] = p2[-2], p1[-2]
        if self.rng.random() < 0.5:
            c1[-1], c2[-1] = p2[-1], p1[-1]
        return c1, c2

    def _mutate(self, ind):
        ind = ind.copy()
        mut_mask = self.rng.random(self.n_weight_genes) < self.p_mut
        ind[:self.n_weight_genes] += mut_mask * self.rng.normal(0, 0.3, self.n_weight_genes)
        if self.rng.random() < self.p_mut:
            ind[-2] = self.rng.integers(0, len(HIDDEN_CHOICES))
        if self.rng.random() < self.p_mut:
            ind[-1] = self.rng.integers(0, len(LR_CHOICES))
        return ind

    def run(self, X_tr, y_tr, X_val, y_val):
        pop = [self._random_individual() for _ in range(self.pop_size)]
        best_overall, best_fit_overall = None, -np.inf

        for gen in range(self.generations):
            fitnesses = np.array([
                self.fitness(ind, X_tr, y_tr, X_val, y_val) for ind in pop
            ])
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_fit_overall:
                best_fit_overall = fitnesses[gen_best_idx]
                best_overall = pop[gen_best_idx].copy()

            self.history_best.append(float(fitnesses.max()))
            self.history_mean.append(float(fitnesses.mean()))

            new_pop = [best_overall.copy()]  # elitism
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, fitnesses)
                p2 = self._tournament_select(pop, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2))
            pop = new_pop

        return best_overall, best_fit_overall

    def build_best_mlp(self, best_individual):
        return self._build_mlp(best_individual)
