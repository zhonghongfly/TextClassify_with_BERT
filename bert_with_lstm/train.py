# coding=utf-8

import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.python.ops.rnn import static_rnn

# 定义LSTM
lstm_cell = BasicLSTMCell(20, forget_bias=1.0)
rnn_outputs, rnn_states = static_rnn(lstm_cell, rnn_input, dtype=tf.float32)

# 利用LSTM最后的输出进行预测
logits = tf.layers.dense(rnn_outputs[-1], num_classes)

predicted_labels = tf.argmax(logits, axis=1)

# 定义损失和优化器
losses= tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(labels_placeholder, num_classes),
    logits=logits
)

mean_loss = tf.reduce_mean(losses)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(mean_loss)

with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())

# 定义要填充的数据
feed_dict = {
    datas_placeholder: datas,
    labels_placeholder: labels
}

print("开始训练")
for step in range(100):
    _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=feed_dict)

    if step % 10 == 0:
        print("step = {}\tmean loss = {}".format(step, mean_loss_val))


print("训练结束，进行预测")
predicted_labels_val = sess.run(predicted_labels, feed_dict=feed_dict)
for i, text in enumerate(all_texts):
    label = predicted_labels_val[i]
    label_name = label_name_dict[label]
    print("{} => {}".format(text, label_name))