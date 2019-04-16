import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

src_pic_path = "../road02_ins/ColorImage/Record001/Camera 5"
depth_pic_path = "../road02_ins_depth/Depth/Record001/Camera 5"

step_num = 10000000
batch_size = 1
shrink = 10

base_learning_rate = 0.0009
examples_num = 200
decay_rate = 0.99
# 初始化权重函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1);
    return tf.Variable(initial, name="weight")


# 初始化偏置项
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="bias")


# 定义卷积函数
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 定义一个2*2的最大池化层
def max_pool_2_2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def deconv2d(x, w, output_shape):
    return tf.nn.conv2d_transpose(value=x, filter=w, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME', name=None)

with tf.name_scope("input"):
    keep_prob = tf.placeholder("float", name="keep_prob")
    x = tf.placeholder(dtype=tf.float32,shape=[batch_size,271,338,3],name="input_data")
    y = tf.placeholder(dtype=tf.float32,shape=[batch_size,271,338],name="input_labels")

with tf.name_scope("conv1"):
    filter = weight_variable([3,3,3,32])
    bias = bias_variable([32])
    conv_res1 = max_pool_2_2(tf.nn.relu(conv2d(x, filter) + bias))
    print(conv_res1.shape)    
    
with tf.name_scope("conv2"):
    filter = weight_variable([3,3,32,64])
    bias = bias_variable([64])
    conv_res2 = max_pool_2_2(tf.nn.relu(conv2d(conv_res1, filter) + bias))
    print(conv_res2.shape) 
    
with tf.name_scope("conv3"):
    filter = weight_variable([3,3,64,128])
    bias = bias_variable([128])
    conv_res3 = max_pool_2_2(tf.nn.relu(conv2d(conv_res2, filter) + bias))
    print(conv_res3.shape)     
    
with tf.name_scope("conv4"):
    filter = weight_variable([3,3,128,256])
    bias = bias_variable([256])
    conv_res4 = max_pool_2_2(tf.nn.relu(conv2d(conv_res3, filter) + bias))
    print(conv_res4.shape) 
    
with tf.name_scope("conv5"):
    filter = weight_variable([3,3,256,512])
    bias = bias_variable([512])
    conv_res5 = max_pool_2_2(tf.nn.relu(conv2d(conv_res4, filter) + bias))
    print(conv_res5.shape)
    
with tf.name_scope("deconv5"):
    filter = weight_variable([3,3,256,512])
    bias = bias_variable([256])
    deconv_res1 = tf.nn.relu(deconv2d(conv_res5, filter, conv_res4.shape) + bias)
    print(deconv_res1.shape)
    
with tf.name_scope("deconv4"):
    filter = weight_variable([3,3,128,256])
    bias = bias_variable([128])
    deconv_res2 = tf.nn.relu(deconv2d(deconv_res1, filter, conv_res3.shape) + bias)
    print(deconv_res2.shape)
    
with tf.name_scope("deconv3"):
    filter = weight_variable([3,3,64,128])
    bias = bias_variable([64])
    deconv_res3 = tf.nn.relu(deconv2d(deconv_res2, filter, conv_res2.shape) + bias)
    print(deconv_res3.shape) 
    
with tf.name_scope("deconv2"):
    filter = weight_variable([3,3,32,64])
    bias = bias_variable([32])
    deconv_res4 = tf.nn.relu(deconv2d(deconv_res3, filter, conv_res1.shape) + bias)
    print(deconv_res4.shape) 
    
with tf.name_scope("deconv1"):
    filter = weight_variable([3,3,3,32])
    bias = bias_variable([3])
    deconv_res5 = tf.nn.relu(deconv2d(deconv_res4, filter, [batch_size, 271, 338, 3]) + bias)
    print(deconv_res5.shape)
    nn_res = tf.reduce_mean(deconv_res5, 3)
    print(nn_res.shape)
    
global_step = tf.Variable(0, trainable=False)
mse = tf.losses.mean_squared_error(nn_res, y) 

learning_rate = tf.train.exponential_decay(base_learning_rate, global_step, examples_num / batch_size, decay_rate, staircase=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(mse,global_step=global_step) #使用adam优化器来以0.0001的学习率来进行微调

saver=tf.train.Saver()
with tf.Session() as sess:    
    init = tf.initialize_all_variables()
    # sess.run(init)
    saver.restore(sess, "./module/test")
    writer = tf.summary.FileWriter("logs/", sess.graph)
    
    data_file_list = os.listdir(src_pic_path)
    label_file_list = os.listdir(depth_pic_path)
    
    print(data_file_list)
    print(label_file_list)
    
    file_num = len(data_file_list)
    print(file_num)

    for insex in range(len(data_file_list)):

    
    for step in range(step_num):
        index = step % file_num
        img_name = src_pic_path + "/" + data_file_list[index]
        label_name = depth_pic_path + "/" + label_file_list[index]
        
        img = cv2.imread(img_name)
        label = cv2.imread(label_name)
       
        img=cv2.resize(img,(int(img.shape[1]/shrink),int(img.shape[0]/shrink)),interpolation=cv2.INTER_CUBIC)
        label=cv2.resize(label,(int(label.shape[1]/shrink),int(label.shape[0]/shrink)),interpolation=cv2.INTER_CUBIC)
        
        img = img[np.newaxis, :]
        label = label[np.newaxis,:,:,0]
        # print(label[0].max(),label[0].min())
        # print(label[:,:,0].shape)
        # cv2.imshow("test", label[:,:,0])
        # cv2.imshow("test2", label[:,:,1])
        # print(label.shape)
        # cv2.imshow("test3", label[0])
        # cv2.waitKey(0)
        
        # plt.subplot(211)
        # plt.imshow(img[0])
        # plt.subplot(212)
        # plt.imshow(label[0])
        # plt.show()
        
        # print(label.shape)
        _, loss, res, lr = sess.run((train_step, mse, nn_res, learning_rate), feed_dict={x:img, y:label})
        if step % 50 == 0:
            saver.save(sess, "module/test")
        # print("step : ")
        
        res = res.reshape((res.shape[1],res.shape[2]))
        res[res<0] = 0
        res = res/res.max()*255
        res = res[np.newaxis,:]
        # print(res.shape)
        print("step : ", step, "   loss : ", loss, "     learning rate : ",lr)

        # print(res.astype(int).max(), res.astype(int).min(),res.shape)
        # plt.imshow(res)
        # plt.show()
        # cv2.imshow("test1", label)
        # cv2.waitKey(0)
        temp = res.astype(int)
        # print("temp",temp.max(),temp.min(),temp.shape)
        # plt.imshow(temp[0])
        # plt.show()
        temp = temp[0]
        cv2.imwrite("test2.jpg",temp)
        # cv2.waitKey(0)
        # print(res.shape)
        
    cv2.waitKey(0)
