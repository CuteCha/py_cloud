import tensorflow as tf
from tensorflow import keras
import numpy as np
import torch


class Transformer4(object):
    def __init__(self, x, num_head, num_block, mask=None):
        self.x = x  # [B,L,d]
        self.mask = mask  # [B,L]
        self.num_head = num_head
        self.num_block = num_block

        self.x_shape = x.get_shape()
        self.x_len = self.x_shape[-2]
        self.x_dim = self.x_shape[-1]
        self.encode_layers = []

    @classmethod
    def make_mask(cls, x_mask):
        '''
        :param x_mask: [B,L]
        :return: mask [B,L,L]
        '''
        B, L = x_mask.get_shape()
        x_mask = tf.cast(tf.reshape(x_mask, [B, 1, L]), tf.float32)
        b_ones = tf.ones(shape=[B, L, 1], dtype=tf.float32)

        return b_ones * x_mask

    def attention(self, q, k, v, mask=None):
        k_t = tf.transpose(k, perm=[0, 2, 1])
        score = tf.matmul(q, k_t) / tf.sqrt(tf.constant(self.x_dim, dtype=tf.float32))

        if mask is not None:
            mask = self.make_mask(mask)
            score += ((1.0 - tf.cast(mask, tf.float32)) * (-1E6))

        score = keras.activations.softmax(score)  # [B,L,L]

        return tf.matmul(score, v)

    def parallel_head(self, x, num, name=None):
        initializer = keras.initializers.GlorotNormal()
        w = tf.Variable(initial_value=initializer(shape=(self.x_dim, self.x_dim * num)), name=name)
        z = tf.matmul(x, w)
        print(f"z.get_shape()={z.get_shape()}")

        return tf.concat(tf.split(z, num, axis=2), axis=0)

    def multi_head(self, x, num):
        q_ = self.parallel_head(x, num, name="w_q")
        k_ = self.parallel_head(x, num, name="w_q")
        v_ = self.parallel_head(x, num, name="w_q")

        o_ = self.attention(q_, k_, v_)  # [B*num,L,d]
        o = tf.concat(tf.split(o_, num, axis=0), axis=2)  # [B,L,num*d]
        print(f"o.get_shape()={o.get_shape()}")

        initializer = keras.initializers.GlorotNormal()
        w_o = tf.Variable(initial_value=initializer(shape=(self.x_dim * num, self.x_dim)), name="w_o")

        return tf.matmul(o, w_o)

    def feed_forward(self, x):
        layer = keras.Sequential([
            keras.layers.Dense(self.x_dim, activation="gelu"),
            keras.layers.Dense(self.x_dim)
        ])

        return layer(x)

    def encode_block(self, x):
        z = self.multi_head(x, self.num_head)
        z = keras.layers.LayerNormalization()(x + z)
        e = self.feed_forward(z)

        return keras.layers.LayerNormalization()(z + e)

    def encoder(self):
        z = self.x
        print(f"z.get_shape()={z.get_shape()}")
        for _ in range(self.num_block):
            e = self.encode_block(z)
            print(f"e.get_shape()={e.get_shape()}")
            self.encode_layers.append(e)
            z = e

    def decoder(self):
        pass


class Transformer5(object):
    def __init__(self, x, y, num_head, num_enc, num_dec, pad_mask=None, look_mask=None):
        self.batch_size, self.x_len, self.x_dim = x.get_shape()
        self.x = x
        self.y = y
        self.num_head = num_head
        self.num_enc = num_enc
        self.num_dec = num_dec
        self.pad_mask = pad_mask
        self.look_mask = look_mask

    def attention(self, q, k, v, mask=None):
        score = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.constant(self.x_dim, dtype=tf.float32))
        if mask is not None:
            score += (tf.cast(mask, dtype=tf.float32) * (-1E6))

        return tf.matmul(score, v)

    def para_head(self, x, num_head):
        initializers = keras.initializers.GlorotNormal()
        w = tf.Variable(initial_value=initializers(self.x_dim, self.x_dim * num_head))
        o_ = tf.matmul(x, w)
        return tf.concat(tf.split(o_, num_head, axis=2), axis=0)

    def multi_head(self, x, num_head, mask):
        q_ = self.para_head(x, num_head)
        k_ = self.para_head(x, num_head)
        v_ = self.para_head(x, num_head)

        o = self.attention(q_, k_, v_, mask)
        o_ = tf.concat(tf.split(o, num_head, axis=0), axis=2)

        initializers = keras.initializers.GlorotNormal()
        w_o = tf.Variable(initial_value=initializers(self.x_dim * num_head, self.x_dim))

        return tf.matmul(o_, w_o)

    def ffn(self, x):
        layer = keras.Sequential([
            keras.layers.Dense(self.x_dim, activation="gelu"),
            keras.layers.Dense(self.x_dim)
        ])

        return layer(x)

    @classmethod
    def ln(cls, x):
        return keras.layers.LayerNormalization()(x)

    def encoder_block(self, x, num_head, mask):
        z = self.multi_head(x, num_head, mask)
        u = self.ln(x + z)
        z = self.ffn(u)
        return self.ln(u + z)

    def cross_multi_head(self, y, enc, num_head, mask):
        q_ = self.para_head(y, num_head)
        k_ = self.para_head(enc, num_head)
        v_ = self.para_head(enc, num_head)

        o_ = self.attention(q_, k_, v_, mask)

        initializers = keras.initializers.GlorotNormal()
        w_o = tf.Variable(initial_value=initializers(self.x_dim * num_head, self.x_dim))

        return tf.matmul(o_, w_o)

    def decoder_block(self, y, enc, num_head, mask):
        z = self.multi_head(y, num_head, mask)
        u = self.ln(y + z)
        z = self.cross_multi_head(u, enc, num_head, mask)
        u = self.ffn(z)
        return self.ln(z + u)


def main():
    x = [[3, 38], [20, 9], [31, 37, 38, 10], [1, 2, 3, 4, 5], [7, 8]]
    x_pad = keras.preprocessing.sequence.pad_sequences(x, maxlen=4, padding="post", truncating="post")
    x_mask = tf.sequence_mask([len(seq) for seq in x], maxlen=4)
    embedding = keras.layers.Embedding(50, 3, input_length=5)
    x_emb = embedding(x_pad)
    # print(x_emb)
    # print("=" * 36)

    trm = Transformer4(x_emb, 6, 2, x_mask)
    trm.encoder()
    print(trm.encode_layers[-1])


def layer_norm(X):
    """
    X->[B,L,d]
    """
    mu = np.mean(X, axis=-1, keepdims=True)  # [B,L,1]
    sigma = np.std(X, axis=-1, keepdims=True)  # [B,L,1]

    alpha = np.ones(X.shape[-1:])  # [d]
    beta = np.zeros(X.shape[-1:])  # [d]
    eps = 1e-6

    res = alpha * (X - mu) / (sigma + eps) + beta

    # print(res)
    return res


def layer_norm_debug():
    np.random.seed(123)
    X = np.random.random([10, 5, 3])
    x0 = layer_norm(X)
    print(x0[0])
    print("-" * 36)
    ln = LayerNorm(X.shape[-1:])
    x1 = ln(torch.from_numpy(X))
    print(x1[0])


class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[-1:],
                                     initializer=tf.ones_initializer(),
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[-1:],
                                    initializer=tf.zeros_initializer(),
                                    trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):  # x shape=[batch_size, seq_len, d_model]
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta


class LayerNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.alpha = torch.nn.Parameter(torch.ones(dim))
        self.beta = torch.nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        """
        x: [B,L,d]
        """
        mean = x.mean(-1, keepdim=True)  # mean: [B, L, 1]
        std = x.std(-1, keepdim=True)  # std: [B, L, 1]
        return self.alpha * (x - mean) / (std + self.eps) + self.beta


if __name__ == '__main__':
    # main()
    layer_norm_debug()
