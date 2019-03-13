import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 忽略AVX AVX2警告

# 节点(nodes)——表示计算单元，也可以是输入的起点或者输出的终点
# 线(edges)——表示节点之间的输入 / 输出关系
# 在 TensorFlow 中，每个节点都是用 tf.Tensor的实例来表示的，即每个节点的输入、输出都是Tensor
# Tensor 即可以表示输入、输出的端点，还可以表示计算单元

# 如下的代码创建了对两个 Tensor 执行 + 操作的 Tensor
# 区分输出1
print("Method1:")
import tensorflow as tf
# 创建两个常量节点
node1 = tf.constant(3.2)
# tf.constant创建的是Tensor的常量，创建后不可更改
node2 = tf.constant(4.8)
# 创建一个 adder 节点，对上面两个节点执行 + 操作
adder = node1 + node2
# 打印一下 adder 节点
print(adder)
# 打印 adder 运行后的结果
sess = tf.Session()
print(sess.run(adder))

# 有时我们还会需要从外部输入数据，这时可以用tf.placeholder 创建占位 Tensor，占位 Tensor 的值可以在运行的时候输入
# 区分输出2
print("Method2:")
import tensorflow as tf
# 创建两个占位 Tensor 节点
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
# 创建一个 adder 节点，对上面两个节点执行 + 操作
adder_node = a + b
# 打印三个节点
print(a)
print(b)
print(adder_node)
# 运行一下，后面的 dict 参数是为占位 Tensor 提供输入数据
sess = tf.Session()
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
# 区分输出3
print("Method3:")
# 添加×操作
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))
