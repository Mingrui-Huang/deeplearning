import tensorflow as tf
import numpy as np


# 导入数据集并转为[0, 1]间的tensor。
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
X_train, X_test = tf.cast(train_x / 255.0, tf.float32), tf.cast(test_x / 255.0, tf.float32)
y_train, y_test = tf.cast(train_y, tf.int16), tf.cast(test_y, tf.int16)
print(X_train)
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

model.fit(X_train, y_train, batch_size=64, epochs=30, validation_split=0.2)

model.evaluate(X_test, y_test, verbose=2)

y_predict = np.argmax(model.predict(X_test), axis=1) 	# 对数据集切片取出前五个样本，并预测其结果

n = 0
for i in range(len(X_test)):
    if y_predict[i] == test_y[i]:
        n = n + 1
accuracy = n / len(X_test)
print('test accuracy: %.3f' % accuracy)
