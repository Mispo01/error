import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import input_data
import numpy as np
import model
import os

# Выберите картинку из указанного каталога
def get_one_image(train):
    files = os.listdir(train)
    n = len(files)
    n = len(train)
    ind = np.random.randint(0,n)
    img_dir = os.path.join(train,files[ind])
    image = Image.open(img_dir)
    plt.imshow(image)
    plt.show()
    image = image.resize([208, 208])
    image = np.array(image)
    return image


# Тестовая картинка
def evaluate_one_image():
    # Сохраните коллекцию картинок, которые мы хотим протестировать
    train = './data/test/'
    image_array = get_one_image(train)


with tf.Graph().as_default():
    BATCH_SIZE = 1  # Поскольку считывается только одна картинка, пакетный режим установлен на 1
    N_CLASSES = 2  ## 2 выходных нейрона, [1, 0] или [0, 1] вероятность кошки и собаки
    # Конвертировать формат изображения
image = tf.cast(image_array, tf.float32)
# Стандартизация изображения
image = tf.image.per_image_standardization(image)
# Картинка изначально трехмерная [208, 208, 3] Переопределить форму рисунка в четырехмерный четырехмерный тензор
image = tf.reshape(image, [1, 208, 208, 3])
logit = model.inference(image, BATCH_SIZE, N_CLASSES)
# Поскольку при возвращении логического вывода не используется функция активации, используйте softmax для активации результата здесь
logit = tf.nn.softmax(logit)
# Используйте самые оригинальные входные данные для ввода данных в заполнитель модели
x = tf.placeholder(tf.float32, shape=[208, 208, 3])

# Путь, где мы храним модель
logs_train_dir = './Logs/train'

# Определить заставку
saver = tf.train.Saver()

with tf.Session() as sess:
    print("Reading checkpoints...")
    # Загрузить модель в sess
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found')

    prediction = sess.run(logit, feed_dict={x: image_array})
    max_index = np.argmax(prediction)
    if max_index == 0:
        print('This is a cat with possibility %.6f' % prediction[:, 0])
    else:
        print('This is a dog with possibility %.6f' % prediction[:, 1])