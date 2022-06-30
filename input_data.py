import tensorflow as tf
import numpy as np
import os


# Получить путь к файлу и метку
def get_files(file_dir):
    # file_dir: путь к папке
    # возврат: картинки и теги после выхода из строя

    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    # Загрузите путь к данным и напишите значение тега
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print("There are %d cats\nThere are %d dogs" % (len(cats), len(dogs)))

    # Перемешать порядок файлов
    image_list = np.hstack((cats, dogs))  # a=[1,2,3] b=[4,5,6] print(np.hstack((a,b)))
    # Выход: [1 2 3 4 5 6]
    label_list = np.hstack((label_cats, label_dogs))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()  # Транспонировать
    np.random.shuffle(temp)  ## Используйте shuffle, чтобы перетасовать заказ

    ## Удалить список из прерванной температуры (img и lab)


image_list = list(temp[:, 0])
label_list = list(temp[:, 1])
label_list = [int(i) for i in label_list]  # Тип строки, преобразованный в тип int

return image_list, label_list


# Генерация пакетов одинакового размера
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # image, label: для создания пакета изображений и списка ярлыков
    # image_W, image_H: ширина и высота изображения
    # batch_size: сколько картинок в каждой партии
    # емкость: емкость очереди, максимальный размер очереди
    # возврат: пакет изображений и тегов

    # Конвертировать тип python.list в формат, распознаваемый tf
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # Создать очередь, поместить изображение и метку в очередь
    input_queue = tf.train.slice_input_producer([image, label])

    image_contents = tf.read_file(input_queue[0])  ## Прочитать всю информацию изображения


label = input_queue[1]
# Декодируйте изображение, различные типы изображений не могут быть смешаны вместе, либо использовать только JPEG или только PNG
## Декодировать картинку, каналы = 3 - это цветная картинка, r, g, b, черно-белая картинка - 1, также можно понимать толщину картинки
image = tf.image.decode_jpeg(image_contents, channels=3)

# Единый размер изображения
# Обрезать или расширить изображение в центре изображения до указанного image_W, image_H
# image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
# Мой метод

image = tf.image.resize_images(image, [image_H, image_W],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # метод интерполяции ближайшего соседа
image = tf.cast(image, tf.float32)  # тип строки, преобразованный в float
# image = tf.image.per_image_standardization (image) # Стандартизировать данные, стандартизация - уменьшить
# Перейти к среднему значению, деленному на его дисперсию

# Генерировать пакеты num_threads Сколько потоков установлено в соответствии с очередностью конфигурации компьютера. Максимальное количество изображений.
# tf.train.shuffle_batch перемешать заказ,
image_batch, label_batch = tf.train.batch([image, label],
                                          batch_size=batch_size,
                                          num_threads=64,  # темы
                                          capacity=capacity)

# Эти две строки избыточны? Переставьте метку, количество строк [batch_size], если вам интересно, вы можете попробовать это
# label_batch = tf.reshape(label_batch, [batch_size])


return image_batch, label_batch

