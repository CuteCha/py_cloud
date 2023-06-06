import numpy as np
import time
import math


class DPPModel(object):
    def __init__(self, **kwargs):
        self.item_count = kwargs['item_count']  # N
        self.item_embed_size = kwargs['item_embed_size']  # M
        self.max_iter = kwargs['max_iter']
        self.epsilon = kwargs['epsilon']

    def build_kernel_matrix(self):
        # 用户和每个item的相关性，排序得分
        rank_score = np.random.random(size=(self.item_count))
        # item的embedding， 这些是提前离线训练出来，实际使用的时候直接读redis。 N*M
        item_embedding = np.random.randn(self.item_count, self.item_embed_size)
        # norm而已，可以一开始在redis中写入归一化后的结果
        item_embedding = item_embedding / np.linalg.norm(item_embedding, axis=1, keepdims=True)
        sim_matrix = np.dot(item_embedding, item_embedding.T)  # item之间的相似度矩阵，N*N
        # 计算kernel矩阵，对角矩阵与其他矩阵的矩阵乘法，等价于对角向量元素相乘目标矩阵。
        self.kernel_matrix = rank_score.reshape((self.item_count, 1)) \
                             * sim_matrix * rank_score.reshape((1, self.item_count))

    def dpp(self):
        # 这里初始化不是空矩阵而是0矩阵，主要是方便论文中ci = [ci, ei]的列扩展。不要迷惑。
        c = np.zeros((self.max_iter, self.item_count))
        d = np.copy(np.diag(self.kernel_matrix))  # L(i,i)
        j = np.argmax(d)  # 这里和论文取对数不一致，因为所有元素大于等于0，取对数后也是递增函数，这里取最大值，无所谓了。
        Yg = [j]  # j已经被加入Yg矩阵中了
        iter = 0
        Z = list(range(self.item_count))
        while len(Yg) < self.max_iter:  # 迭代停止条件
            # 差集
            Z_Y = set(Z).difference(set(Yg))
            for i in Z_Y:
                # 这里单独区分iter=0，因为在某些python低版本中做减法的部分会出错，因为是空矩阵，python3.7不会，所以其实可以合并。
                if iter == 0:
                    ei = self.kernel_matrix[j, i] / np.sqrt(d[j])
                else:
                    ei = (self.kernel_matrix[j, i] - np.dot(c[:iter, j], c[:iter, i])) / np.sqrt(d[j])
                c[iter, i] = ei
                d[i] = d[i] - ei * ei
            d[j] = 0  # 保证j不会被再次选择到，否则Yg内有两个一样，意味着排序列表中有两个相同的物料，显然不符合实际要求。
            j = np.argmax(d)
            if d[j] < self.epsilon:
                break
            Yg.append(j)
            iter += 1

        return Yg  # 新的物料排序结果


def main():
    kwargs = {
        'item_count': 50,
        'item_embed_size': 16,
        'max_iter': 200,
        'epsilon': 0.01
    }
    start = time.time()
    dpp_model = DPPModel(**kwargs)
    dpp_model.build_kernel_matrix()
    item_list = dpp_model.dpp()
    print(item_list)
    end = time.time()
    print("耗时：{}".format(end - start))
    print("长度：{}".format(len(item_list)))


if __name__ == "__main__":
    main()
