import tensorflow as tf


# Структура
# conv1 Сверточный слой 1
# pooling1_lrn Слой пула 1
# conv2 Сверточный слой 2
# pooling2_lrn Слой пула 2
# local3 Полностью связанный слой 1
# local4 Полностью связанный слой 2
# softmax полностью подключенный слой 3

def inference(images, batch_size, n_classes):
    # conv1, shape = [kernel_size, kernel_size, channels, kernel_numbers]
    # Сверточный слой 1 16 Ядра свертки 3x3 (3 канала), заполнение = 'SAME', указывающее, что размер свернутого изображения после заполнения совпадает с исходным изображением, а функция активации relu ()
    with tf.variable_scope("conv1") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name="conv1")

    # pool1 && norm1
    # Уровень пула 1 Максимальный пул 3x3, шаг шага 2, после пула выполните операцию lrn (), локальный ответ нормализуется, что полезно для обучения.
    with tf.variable_scope("pooling1_lrn") as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling1")
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm1')

    # conv2
    # Сверточный слой 2 16 Ядра свертки 3x3 (16 каналов), заполнение = 'SAME', указывающее, что размер свернутого изображения после заполнения совпадает с исходным изображением, а функция активации relu ()
    with tf.variable_scope("conv2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name="conv2")

    # pool2 && norm2
    # Уровень пула 2 Максимальный пул 3x3, шаг шага 2, выполнить операцию lrn () после объединения,
    with tf.variable_scope("pooling2_lrn") as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling2")
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm2')

    # full-connect1
    # Полностью подключенный слой 1 128 нейронов, преобразовать вывод предыдущего слоя пула в линию и активировать функцию relu ()
    with tf.variable_scope("fc1") as scope:
        reshape = tf.reshape(norm2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable("weights",
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name="fc1")

    # full_connect2
    # Полностью связанный слой 2 128 нейронов, функция активации relu ()
    with tf.variable_scope("fc2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name="fc2")

    # softmax
    # Softmax регрессионный слой Выведите предыдущий слой FC, выполните линейную регрессию, рассчитайте оценку для каждой категории, здесь есть 2 категории, поэтому этот слой выводит две оценки
    with tf.variable_scope("softmax_linear") as scope:
        # weights = tf.get_variable ("softmax_linear", есть ссылка, написанная так, вы можете попробовать
        weights = tf.get_variable("weights",
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name="softmax_linear")
        softmax_linear = tf.nn.softmax(softmax_linear)
    return softmax_linear


# расчет потерь
def losses(logits, labels):
    with tf.variable_scope("loss") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name="xentropy_per_example")
        loss = tf.reduce_mean(cross_entropy, name="loss")
        tf.summary.scalar(scope.name + "loss", loss)
    return loss

оптимизация значения потерь
def trainning(loss, learning_rate):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

# Оценка / расчет точности
def evaluation(logits, labels):
    with tf.variable_scope("accuracy") as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + "accuracy", accuracy)
    return accuracy
