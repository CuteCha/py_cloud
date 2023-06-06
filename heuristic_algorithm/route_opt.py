import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import heapq
import itertools

'''
V0=1
lambda=1
vs=1
'''


def get_position(num):
    np.random.seed(123)
    return np.concatenate([np.array([[0.0, 0.0]]), np.random.random((num, 2))], axis=0)


class BruteOpt(object):
    def __init__(self, position):
        self.position = position
        self.num_point = len(self.position) - 1
        self.dis_matrix = self.cal_dis_matrix()
        self.best_individual = None

    def cal_dis_matrix(self):
        a = np.tile(np.sum(self.position ** 2, axis=1, keepdims=True), self.num_point + 1)
        b = np.transpose(a)
        c = np.matmul(self.position, np.transpose(self.position))

        return np.sqrt(a + b - 2 * c)

    def cal_fitness(self, perm):
        dis_lst = np.array([self.dis_matrix[perm[i]][perm[i + 1]] for i in range(len(perm) - 1)])
        cum_time = np.cumsum(dis_lst)

        fitness = np.sum(np.exp(-cum_time))

        return fitness

    def gen_all_permutation(self):
        arr = [i for i in range(1, self.num_point + 1)]
        res = list(itertools.permutations(arr))

        return res

    def run(self):
        best_fitness = 0.0
        for perm in self.gen_all_permutation():
            perm = [0] + list(perm)
            fitness = self.cal_fitness(perm)
            if fitness > best_fitness:
                best_fitness = fitness
                self.best_individual = Individual(perm, fitness)


class Individual(object):
    def __init__(self, perm, fitness):
        self.perm = perm
        self.fitness = fitness


class GenOpt(object):
    def __init__(self, position, population_num=100, pc=0.9, pm=0.1, max_iter=100):
        self.population_num = population_num
        self.pc = pc
        self.pm = pm
        self.max_iter = max_iter
        self.position = position
        self.num_point = len(self.position) - 1
        self.dis_matrix = self.cal_dis_matrix()
        self.best_individual = None
        self.individual_lst = []
        self.his_best_individual_lst = []

    def cal_dis_matrix(self):
        a = np.tile(np.sum(self.position ** 2, axis=1, keepdims=True), self.num_point + 1)
        b = np.transpose(a)
        c = np.matmul(self.position, np.transpose(self.position))

        return np.sqrt(a + b - 2 * c)

    def cal_fitness(self, perm):
        dis_lst = np.array([self.dis_matrix[perm[i]][perm[i + 1]] for i in range(len(perm) - 1)])
        cum_time = np.cumsum(dis_lst)

        fitness = np.sum(np.exp(-cum_time))

        return fitness

    def cross(self):
        new_individual_lst = []
        random.shuffle(self.individual_lst)
        for i in range(0, self.population_num - 1, 2):
            perm1 = copy.deepcopy(self.individual_lst[i].perm)
            perm2 = copy.deepcopy(self.individual_lst[i + 1].perm)

            idx1 = random.randint(1, self.num_point - 1)
            idx2 = random.randint(idx1, self.num_point)

            record1 = {val: idx for idx, val in enumerate(perm1)}
            record2 = {val: idx for idx, val in enumerate(perm2)}

            for j in range(idx1, idx2):
                val1, val2 = perm1[j], perm2[j]
                pos1, pos2 = record1[val2], record2[val1]
                perm1[j], perm1[pos1] = perm1[pos1], perm1[j]
                perm2[j], perm2[pos2] = perm2[pos2], perm2[j]
                record1[val1], record1[val2] = pos1, j
                record2[val1], record2[val2] = j, pos2

            new_individual_lst.append(Individual(perm1, self.cal_fitness(perm1)))
            new_individual_lst.append(Individual(perm2, self.cal_fitness(perm2)))

        return new_individual_lst

    def mutate(self, individual_lst):
        for individual in individual_lst:
            if random.random() < self.pm:
                ori_perm = copy.deepcopy(individual.perm)
                idx1 = random.randint(1, self.num_point - 1)
                idx2 = random.randint(idx1, self.num_point)
                m_perm = ori_perm[idx1:idx2]
                m_perm.reverse()
                new_perm = ori_perm[:idx1] + m_perm + ori_perm[idx2:]
                individual = Individual(new_perm, self.cal_fitness(new_perm))

            self.individual_lst.append(individual)

    def select(self):
        heap = []
        for individual in self.individual_lst:
            fitness = individual.fitness
            if fitness == 0:
                continue
            u = random.uniform(0, 1)
            k = u ** (1 / fitness)

            if len(heap) < self.population_num:
                heapq.heappush(heap, (k, individual))
            elif k > heap[0][0]:
                heapq.heappush(heap, (k, individual))
                if len(heap) > self.population_num:
                    heapq.heappop(heap)

        self.individual_lst = [each[1] for each in heap]

    def evolution(self):
        new_individual_lst = self.cross()
        self.mutate(new_individual_lst)
        self.select()

        for individual in self.individual_lst:
            if individual.fitness > self.best_individual.fitness:
                self.best_individual = copy.deepcopy(individual)

    def gen_perm(self):
        perm = [i for i in range(1, self.num_point + 1)]
        random.shuffle(perm)
        return [0] + perm

    def run(self):
        for _ in range(self.population_num):
            perm = self.gen_perm()
            self.individual_lst.append(Individual(perm, self.cal_fitness(perm)))

        self.best_individual = copy.deepcopy(self.individual_lst[0])

        for i in range(self.max_iter):
            self.evolution()
            result = copy.deepcopy(self.best_individual)
            self.his_best_individual_lst.append(result)

    def show_route(self):
        perm = self.best_individual.perm
        x = [self.position[i][0] for i in perm]
        y = [self.position[i][1] for i in perm]

        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, c='r')

        n = len(x)
        for i in range(n):
            plt.annotate(i, (x[i], y[i]))

        for i in range(n - 1):
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
            plt.quiver(x[i], y[i], dx, dy, angles='xy', scale=1.03, scale_units='xy', width=0.005, color='b')

        plt.title("opt route")
        plt.show()

    def show_his_fitness(self):
        plt.figure(figsize=(6, 6))
        plt.plot([ind.fitness for ind in self.his_best_individual_lst])
        plt.title("history fitness")
        plt.show()


def main():
    position = get_position(5)

    brute_opt = BruteOpt(position)
    brute_opt.run()
    print(f"{brute_opt.best_individual.fitness}\n{brute_opt.best_individual.perm}")

    print("=" * 36)
    gen_opt = GenOpt(position)
    gen_opt.run()
    print(f"{gen_opt.best_individual.fitness}\n{gen_opt.best_individual.perm}")
    gen_opt.show_route()
    gen_opt.show_his_fitness()


if __name__ == '__main__':
    main()
