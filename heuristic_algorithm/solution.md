#问题建模

将坐标原点$O$标记为第$0$个点，仓库$A$、$B$、$C$、$D$、$E$依次标记为第$1$、$2$、$3$、$4$、$5$个点。
用$d_{ij}$表示第$i$个点到第$j$个点的距离，卡车的速度$V_s$已知，
可以求出卡车从第$i$个点到第$j$个点的所需要的时间$t_{ij}$.

- 1.当卡车容量无限和里程无限时，每个仓库仅可经过一次，
最大化卡车从这些仓库载入的库存总量的卡车行驶的顺序路径是下列问题的最优解：

$$ \max_{i_1,i_2,...,i_5} \sum_{k=1}^{5} V_0\exp^{-\lambda\sum_{l=1}^{k} t_{i_{l-1}i_l}}$$

  $\qquad$其中，$i_1,i_2,...,i_5$是$1,2,...,5$的一个全排列。

- 2.当卡车容量无限和里程无限，仓库仅有$2$个时，每个仓库仅可经过一次，
最大化卡车从这些仓库载入的库存总量的卡车行驶的顺序路径是下列问题的的可行优解就两个$0\rightarrow1\rightarrow2$、$0\rightarrow2\rightarrow1$。

   容易得到：$0\rightarrow1\rightarrow2$比$0\rightarrow2\rightarrow1$更优等价于$t_{01}\le t_{02}$


- 3.当卡车里程为W时，对应下列问题的最优解：
$$ \max_{i_1,i_2,...,i_m} \sum_{k=1}^{m} V_0\exp^{-\lambda\sum_{l=1}^{k} t_{i_{l-1}i_l}}$$
  $$ s.t. \sum_{k=1}^{m} d_{i_{k-1}i_k} \le W$$
$\qquad$其中，$i_1,i_2,...,i_m$ 是 $1,2,...,5$的一个排列。

#求解结果
当仓库个数比较大时，该问题不容易快速得到最优解，通常采用启发式算法求解，这里使用遗传算法进行求解，该算法可以扩展到任意多个仓库，详细过程见代码。
代码包含暴力求解和遗传算法求解两种方法，多次实验，在n不太大的时候遗传算法可以快速得到最优解。
结果如下：

| 实验        |                                                   里程无限 |                      里程有限                      |
| --------   |-------------------------------------------------------:|:----------------------------------------------:|
| 最优路径     |                          ![](file:////path/route2.png) |         ![](file:////path/route1.png)          |
| 收敛过程        |         ![](file:////path/fitness2.png) | ![](file:////path/fitness1.png) |
| 收敛结果        | 总量：2.7006170966053333<br>(路径：[0, 6, 5, 2, 3, 4, 1, 7]) | 总量：2.2417466061262297<br>(路径：[0, 6, 5, 2, 3])  |

#代码
```python
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
w=1.5
'''


def get_position(num, seed=123):
    np.random.seed(seed)
    return np.concatenate([np.array([[0.0, 0.0]]), np.random.random((num, 2))], axis=0)


class BruteOpt(object):
    def __init__(self, position, max_mileage=None):
        self.position = position
        self.max_mileage = max_mileage
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
        n = len(cum_time)

        if self.max_mileage is None:
            fitness = np.sum(np.exp(-cum_time))
            return fitness, n

        else:
            fitness = 0.0
            idx = 0
            for i in range(n):
                if cum_time[i] > self.max_mileage:
                    idx = i
                    break
                fitness += np.exp(-cum_time[i])
            return fitness, idx

    def gen_all_permutation(self):
        arr = [i for i in range(1, self.num_point + 1)]
        res = list(itertools.permutations(arr))

        return res

    def run(self):
        best_fitness = 0.0
        for perm in self.gen_all_permutation():
            perm = [0] + list(perm)
            fitness, idx = self.cal_fitness(perm)
            if fitness > best_fitness:
                best_fitness = fitness
                self.best_individual = Individual(perm, fitness, idx)

    def print_route(self):
        print(f"route={self.best_individual.perm[:self.best_individual.idx + 1]}"
              f"\nfitness={self.best_individual.fitness}")


class Individual(object):
    def __init__(self, perm, fitness, idx):
        self.perm = perm
        self.fitness = fitness
        self.idx = idx


class GenOpt(object):
    def __init__(self, position, max_mileage=None, population_num=100, pc=0.9, pm=0.1, max_iter=100):
        self.population_num = population_num
        self.pc = pc
        self.pm = pm
        self.max_iter = max_iter
        self.position = position
        self.num_point = len(self.position) - 1
        self.dis_matrix = self.cal_dis_matrix()
        self.best_individual = None
        self.max_mileage = max_mileage
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
        n = len(cum_time)

        if self.max_mileage is None:
            fitness = np.sum(np.exp(-cum_time))
            return fitness, n

        else:
            fitness = 0.0
            idx = 0
            for i in range(n):
                if cum_time[i] > self.max_mileage:
                    idx = i
                    break
                fitness += np.exp(-cum_time[i])
            return fitness, idx

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

            fitness1, f_idx1 = self.cal_fitness(perm1)
            new_individual_lst.append(Individual(perm1, fitness1, f_idx1))
            fitness2, f_idx2 = self.cal_fitness(perm2)
            new_individual_lst.append(Individual(perm2, fitness2, f_idx2))

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
                fitness, f_idx = self.cal_fitness(new_perm)
                individual = Individual(new_perm, fitness, f_idx)

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
            fitness, f_idx = self.cal_fitness(perm)
            self.individual_lst.append(Individual(perm, fitness, f_idx))

        self.best_individual = copy.deepcopy(self.individual_lst[0])

        for i in range(self.max_iter):
            self.evolution()
            result = copy.deepcopy(self.best_individual)
            self.his_best_individual_lst.append(result)

    def print_route(self):
        print(f"route={self.best_individual.perm[:self.best_individual.idx + 1]}"
              f"\nfitness={self.best_individual.fitness}")

    def show_route(self):
        perm = self.best_individual.perm
        x = [self.position[i][0] for i in perm]
        y = [self.position[i][1] for i in perm]

        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, c='r')

        n = self.best_individual.idx + 1
        for i in range(n):
            plt.annotate(f"P{perm[i]}(R{i})", (x[i], y[i]))

        for i in range(n - 1):
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
            plt.quiver(x[i], y[i], dx, dy, angles='xy', scale=1.03, scale_units='xy', width=0.005, color='b')

        plt.title(f"opt route(n={self.num_point},W={self.max_mileage})")
        plt.show()

    def show_his_fitness(self):
        plt.figure(figsize=(6, 6))
        plt.plot([ind.fitness for ind in self.his_best_individual_lst])
        plt.title(f"history fitness(n={self.num_point},W={self.max_mileage})")
        plt.show()


def main():
    position = get_position(7, 97)
    mileage = None  # 1.5

    brute_opt = BruteOpt(position, mileage)
    brute_opt.run()
    brute_opt.print_route()

    print("=" * 36)
    gen_opt = GenOpt(position, mileage)
    gen_opt.run()
    gen_opt.print_route()
    gen_opt.show_route()
    gen_opt.show_his_fitness()


if __name__ == '__main__':
    main()

```