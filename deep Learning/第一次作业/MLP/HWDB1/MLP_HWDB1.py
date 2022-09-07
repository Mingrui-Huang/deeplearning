# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import io
import os


def preprocess_image(image):
    image_size = 28
    img_tensor = tf.image.decode_jpeg(image, channels=1)
    img_tensor = tf.image.resize(img_tensor, [image_size, image_size])
    img_tensor /= 255.0  # normalize to [0,1] range
    return img_tensor

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def load_HWDB1():
    root_path = './DATA/HWDB1'
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    temp = []
    with open(os.path.join(root_path, 'train.txt'), 'r') as f:
        for line in f:
            line = line.strip('\n')
            if line != '':
                imgpath = line[:-2]
                label = line[-1:]
                train_images.append(imgpath)
                train_labels.append(int(label))
    for item in train_images:
        img = load_and_preprocess_image(item)
        temp.append(img)
    x_train = np.array(temp)
    y_train = np.array(train_labels)

    temp = []
    with open(os.path.join(root_path, 'test.txt'), 'r') as f:
        for line in f:
            line = line.strip('\n')
            if line != '':
                imgpath = line[:-2]
                label = line[-1:]
                test_images.append(imgpath)
                test_labels.append(int(label))
    for item in test_images:
        img = load_and_preprocess_image(item)
        temp.append(img)
    x_test = np.array(temp)
    y_test = np.array(test_labels)
    return (x_train, y_train), (x_test, y_test)


(train_x, train_y), (test_x, test_y) = load_HWDB1()
X_train, X_test = tf.cast(train_x / 255.0, tf.float32), tf.cast(test_x / 255.0, tf.float32)
y_train, y_test = tf.cast(train_y, tf.int16), tf.cast(test_y, tf.int16)


# 搭建MLP
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # 变成一维数组
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
# 打印模型
model.summary()

# 设置优化器
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

model.fit(X_train, y_train, batch_size=64, epochs=200, validation_split=0.2)
model.evaluate(X_test, y_test, verbose=2)
y_predict = np.argmax(model.predict(X_test), axis=1)


n = 0
for i in range(len(X_test)):
    if y_predict[i] == test_y[i]:
        n = n + 1
accuracy = n / len(X_test)
print('test accuracy: %.3f' % accuracy)

