import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 忽略AVX AVX2警告
# 梯度下降(Gradient Descent)算法

#   建立模型
#    输入     1        2        3        6         8
#    输出    4.8      8.5     10.4       21       25.3
# 损失模型 loss = sum(y - y')~2
# 线性模型 y = Wx + b 中，输入x可以用占位 Tensor 表示，
# 输出y可以用线性模型的输出表示，我们需要不断的改变W和b的值，来找到一个使loss最小的值。
# 这里W和b可以用变量 Tensor 表示。
# 使用tf.Variable()可以创建一个变量 Tensor。

import tensorflow as tf
# 创建变量 W 和 b 节点，并设置初始值
W = tf.Variable([.1], dtype=tf.float32)
b = tf.Variable([-.1], dtype=tf.float32)
# 创建 x 节点，用来输入实验中的输入数据
x = tf.placeholder(tf.float32)
# 创建线性模型
linear_model = W*x + b
# 创建 y 节点，用来输入实验中得到的输出数据，用于损失模型计算
y = tf.placeholder(tf.float32)
# 创建损失模型
loss = tf.reduce_sum(tf.square(linear_model - y))
# 创建 Session 用来计算模型
sess = tf.Session()
# 初始化变量    变量 Tensor 需要经过下面的 init 过程后才能使用
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(W))
# 变量初始化完之后，我们可以先用上面对W和b设置的初始值0.1和-0.1运行一下我们的线性模型看看结果:
print(sess.run(linear_model, {x: [1, 2, 3, 6, 8]}))
# 损失模型
print(sess.run(loss, {x: [1, 2, 3, 6, 8], y: [4.8, 8.5, 10.4, 21.0, 25.3]}))

# 可以用 tf.assign() 对W和b变量重新赋值再检验一下
# 给 W 和 b 赋新值
fixW = tf.assign(W, [2.])
fixb = tf.assign(b, [1.])
# run 之后新值才会生效
sess.run([fixW, fixb])
# 重新验证损失值
print(sess.run(loss, {x: [1, 2, 3, 6, 8], y: [4.8, 8.5, 10.4, 21.0, 25.3]}))

# Conclusion: 我们需要不断调整变量W和b的值，找到使损失值最小的W和b。接下来使用 TensorFlow 进行训练模型。
# 梯度下降(Gradient Descent)算法

# 创建一个梯度下降优化器，学习率为0.001
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)
# 用两个数组保存训练数据
x_train = [1, 2, 3, 6, 8]
y_train = [4.8, 8.5, 10.4, 21.0, 25.3]
# 训练10000次
for i in range(10000):
    sess.run(train, {x: x_train, y: y_train})
# 打印一下训练后的结果
print('W: %s b: %s loss: %s' % (sess.run(W), sess.run(b), sess.run(loss, {x: x_train , y: y_train})))

