# -*- encoding:utf-8 -*-

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras

BUFFER_SIZE = 10  # Use a much larger value for real code
BATCH_SIZE = 64
NUM_EPOCHS = 5
STEPS_PER_EPOCH = 5

datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = datasets['train'], datasets['test']


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label


train_data = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_data = mnist_test.map(scale).batch(BATCH_SIZE)

train_data = train_data.take(STEPS_PER_EPOCH)
test_data = test_data.take(STEPS_PER_EPOCH)

image_batch, label_batch = next(iter(train_data))

# keras train flow
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.02),
                        input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10)
])

# Model is the full model w/o custom layers
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_data, epochs=NUM_EPOCHS)
loss, acc = model.evaluate(test_data)

print("Loss {}, Accuracy {}".format(loss, acc))

# Customize train flow
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.02),
                        input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10)
])

optimizer = keras.optimizers.Adam(0.001)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)


@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        regularization_loss = tf.math.add_n(model.losses)
        pred_loss = loss_fn(labels, predictions)
        total_loss = pred_loss + regularization_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


for epoch in range(NUM_EPOCHS):
    for inputs, labels in train_data:
        train_step(inputs, labels)
    print("Finished epoch", epoch)

keras.layers.Embedding()


# 搭建双任务并训练
def get_model():
    """函数式API搭建Multi_task模型"""

    # 输入
    user_id = keras.layers.Input(shape=(1,), name="user_id")
    store_id = keras.layers.Input(shape=(1,), name="store_id")
    sku_id = keras.layers.Input(shape=(1,), name="sku_id")
    search_keyword = keras.layers.Input(shape=(1,), name="search_keyword")
    category_id = keras.layers.Input(shape=(1,), name="category_id")
    brand_id = keras.layers.Input(shape=(1,), name="brand_id")
    ware_type = keras.layers.Input(shape=(1,), name="ware_type")

    # user特征
    user_vector = keras.layers.concatenate([
        keras.layers.Embedding(num_user_ids, 32)(user_id),
        keras.layers.Embedding(num_store_ids, 8)(store_id),
        keras.layers.Embedding(num_search_keywords, 16)(search_keyword)
    ])
    user_vector = keras.layers.Dense(32, activation='relu')(user_vector)
    user_vector = keras.layers.Dense(8, activation='relu',
                                     name="user_embedding", kernel_regularizer='l2')(user_vector)

    # item特征
    movie_vector = keras.layers.concatenate([
        keras.layers.Embedding(num_sku_ids, 32)(sku_id),
        keras.layers.Embedding(num_category_ids, 8)(category_id),
        keras.layers.Embedding(num_brand_ids, 8)(brand_id),
        keras.layers.Embedding(num_ware_types, 2)(ware_type)
    ])
    movie_vector = keras.layers.Dense(32, activation='relu')(movie_vector)
    movie_vector = keras.layers.Dense(8, activation='relu',
                                      name="movie_embedding", kernel_regularizer='l2')(movie_vector)

    x = keras.layers.concatenate([user_vector, movie_vector])
    out1 = keras.layers.Dense(16, activation='relu')(x)
    out1 = keras.layers.Dense(8, activation='relu')(out1)
    out1 = keras.layers.Dense(1, activation='sigmoid', name='out1')(out1)

    out2 = keras.layers.Dense(16, activation='relu')(x)
    out2 = keras.layers.Dense(8, activation='relu')(out2)
    out2 = keras.layers.Dense(1, activation='sigmoid', name='out2')(out2)

    return keras.models.Model(inputs=[user_id, sku_id, store_id, search_keyword, category_id, brand_id, ware_type],
                              outputs=[out1, out2])
