import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

all_images = np.loadtxt('mnist_train.csv',delimiter=',')[:,1:]
n_nodes_inpl = 784  
n_nodes_hl  = 32  
n_nodes_outl = 784  
# 设定隐层与输出层 weights，biases
hidden_layer_vals = {
'weights':tf.Variable(tf.random_normal([n_nodes_inpl,n_nodes_hl])),
'biases':tf.Variable(tf.random_normal([n_nodes_hl]))  }
output_layer_vals = {
'weights':tf.Variable(tf.random_normal([n_nodes_hl,n_nodes_outl])),   
'biases':tf.Variable(tf.random_normal([n_nodes_outl])) }
# 图片输入
input_layer = tf.placeholder('float', [None, 784])
# 隐层的输入
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_layer,hidden_layer_vals['weights']),hidden_layer_vals['biases']))
# 输出层的输入
output_layer = tf.matmul(layer_1,output_layer_vals['weights']) + output_layer_vals['biases']
# 原始图片
output_true = tf.placeholder('float', [None, 784])
# 定义cost function
meansq = tf.reduce_mean(tf.square(output_layer - output_true))
# 定义 optimizer
learn_rate = 0.1   
optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(meansq)
# 初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100    # 批处理数目
hm_epochs =1000     # 迭代次数
tot_images = 60000  # 图片总数

for epoch in range(hm_epochs):
    epoch_loss = 0    
    for i in range(int(tot_images/batch_size)):
        epoch_x = all_images[ i*batch_size : (i+1)*batch_size ]
        _, c=sess.run([optimizer, meansq], feed_dict={input_layer: epoch_x, output_true: epoch_x})
        epoch_loss+= c
    print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)
# 测试图片
all_images = np.loadtxt('mnist_test.csv',delimiter=',')[:,1:]
any_image = all_images[1000]
output_any_image = sess.run(output_layer, feed_dict={input_layer:[any_image]})
encoded_any_image = sess.run(layer_1, feed_dict={input_layer:[any_image]})
# 显示图片
plt.imshow(any_image.reshape(28,28),  cmap='Greys')
plt.show()
print(encoded_any_image)
plt.imshow(output_any_image.reshape(28,28),  cmap='Greys')
plt.show()
