import os
import numpy as np
import tensorflow as tf
import input_data
import model

N_CLASSES = 2 # 2 выходных нейрона, [1, 0] или [0, 1] вероятность кошки и собаки
 IMG_W = 208 # Изменение размера изображения, если оно слишком большое, время тренировки велико
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 2000
 MAX_STEP = 10000 # Обычно больше 10K
 стоимость обучения = 0,0001 # обычно меньше 0,0001

train_dir = './data/train/'
logs_train_dir = './logs/train/'  # Этот каталог будет создан автоматически

# Получить фотографии и наборы этикеток
train, train_label = input_data.get_files(train_dir)
## Создать партию
train_batch, train_label_batch = input_data.get_batch(train,
                                                      train_label,
                                                      IMG_W,
                                                      IMG_H,
                                                      BATCH_SIZE,
                                                      CAPACITY)

# Оперативное определение Введите модель
train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
# Получить потери
train_loss = model.losses(train_logits, train_label_batch)
# Подготовка
train_op = model.trainning(train_loss, learning_rate)
# Получить точность
train__acc = model.evaluation(train_logits, train_label_batch)
Резюме
слияния
summary_op = tf.summary.merge_all()  # Это сводная запись журнала
# Создать разговор
sess = tf.Session()
# Генерация писателя для записи файлов журнала
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
# Создайте заставку для хранения обученной модели
saver = tf.train.Saver()
# Инициализация всех узлов
sess.run(tf.global_variables_initializer())

# Мониторинг очереди
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# Выполнить пакетное обучение
try:
    # Выполните шаг обучения MAX_STEP, по одной партии за раз
    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break
            # Запустите следующий узел операции, возникает вопрос, почему train_logits здесь не включен?
        _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
        # Распечатывать текущую потерю и получать каждые 50 шагов, записывать журнал и писать в писатель
        if step % 50 == 0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)
            # Сохраняйте обученную модель каждые 2000 шагов
        if step % 2000 == 0 or (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()