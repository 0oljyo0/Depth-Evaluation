import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import math


batch_size = 30
input_size = [batch_size,271,338,3]
output_size = [batch_size,271,338]
step_num = 10000000
shrink = 10

base_learning_rate = 0.0009
examples_num = 200
decay_rate = 0.99

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1);
    return tf.Variable(initial, name="weight")

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="bias")

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2_2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def deconv2d(x, w, output_shape):
    return tf.nn.conv2d_transpose(value=x, filter=w, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME', name=None)

def equal_res_block(block_input,filters):
    weights = weight_variable([1,1,block_input.shape[3].value,filters[0]])
    res_temp = conv2d(block_input,weights)
    res_temp = tf.layers.batch_normalization(res_temp,training=True)
    res_temp = tf.nn.relu(res_temp)

    weights = weight_variable([3,3,64,filters[1]])
    res_temp = conv2d(res_temp,weights)
    res_temp = tf.layers.batch_normalization(res_temp,training=True)
    res_temp = tf.nn.relu(res_temp)
    
    weights = weight_variable([1,1,filters[1],block_input.shape[3].value])
    res_temp = conv2d(res_temp,weights)
    res_temp = tf.layers.batch_normalization(res_temp,training=True)
    
    output = tf.nn.relu(res_temp + block_input)
    
    return output

input_shape4d = input_size
output_shape4d = output_size

with tf.name_scope(name="input"):
    x = tf.placeholder(dtype=tf.float32, shape=input_shape4d, name="input_x")
    y = tf.placeholder(dtype=tf.float32, shape=output_shape4d, name="input_y")

with tf.name_scope(name="conv1"):
    weights = weight_variable([3, 3, 3, 32])
    bias = bias_variable([32])
    conv_temp = conv2d(x,weights) + bias
    conv_temp = tf.layers.batch_normalization(conv_temp,training=True)
    conv1_output = max_pool_2_2(tf.nn.relu(conv_temp))

    print("conv1 output : " , conv1_output.shape)


with tf.name_scope(name="conv2"):
    weights = weight_variable([3, 3, 32, 64])
    bias = bias_variable([64])
    conv_temp = conv2d(conv1_output,weights) + bias
    conv_temp = tf.layers.batch_normalization(conv_temp,training=True)
    conv2_output = max_pool_2_2(tf.nn.relu(conv_temp))

    print("conv2 output : " , conv2_output.shape)

with tf.name_scope(name="conv3"):
    weights = weight_variable([3, 3, 64, 128])
    bias = bias_variable([128])
    conv_temp = conv2d(conv2_output,weights) + bias
    conv_temp = tf.layers.batch_normalization(conv_temp,training=True)
    conv3_output = max_pool_2_2(tf.nn.relu(conv_temp))

    print("conv3 output : " , conv3_output.shape)

with tf.name_scope(name="conv4"):
    weights = weight_variable([3, 3, 128, 256])
    bias = bias_variable([256])
    conv_temp = conv2d(conv3_output,weights) + bias
    conv_temp = tf.layers.batch_normalization(conv_temp,training=True)
    conv4_output = max_pool_2_2(tf.nn.relu(conv_temp))
    
    print("conv4 output : " , conv4_output.shape)

with tf.name_scope(name="res1"):
    res1_output = equal_res_block(conv4_output,[64,128])
    print("res1 output : ",res1_output.shape)
with tf.name_scope(name="res2"):
    res2_output = equal_res_block(res1_output,[64,128])
    print("res2 output : ",res2_output.shape)
with tf.name_scope(name="res3"):
    res3_output = equal_res_block(res2_output,[64,128])
    print("res3 output : ",res3_output.shape)
with tf.name_scope(name="res4"):
    res4_output = equal_res_block(res3_output,[64,128])
    print("res4 output : ",res4_output.shape)
with tf.name_scope(name="res5"):
    res5_output = equal_res_block(res4_output,[64,128])
    print("res5 output : ",res5_output.shape)
with tf.name_scope(name="res6"):
    res6_output = equal_res_block(res5_output,[64,128])
    print("res6 output : ",res6_output.shape)
with tf.name_scope(name="res7"):
    res7_output = equal_res_block(res6_output,[64,128])
    print("res7 output : ",res7_output.shape)
with tf.name_scope(name="res8"):
    res8_output = equal_res_block(res7_output,[64,128])
    print("res8 output : ",res8_output.shape)


with tf.name_scope("deconv4"):
    filter = weight_variable([3,3,128,256])
    bias = bias_variable([128])
    deconv_res2 = tf.nn.relu(deconv2d(res8_output, filter, conv3_output.shape) + bias)
    # deconv_res2 = tf.layers.batch_normalization(conv_temp,training=True)
    print("deconv4 output : ",deconv_res2.shape)

with tf.name_scope("deconv3"):
    filter = weight_variable([3,3,64,128])
    bias = bias_variable([64])
    deconv_res3 = tf.nn.relu(deconv2d(deconv_res2, filter, conv2_output.shape) + bias)
    # deconv_res3 = tf.layers.batch_normalization(conv_temp,training=True)
    print("deconv3 output : ",deconv_res3.shape) 

with tf.name_scope("deconv2"):
    filter = weight_variable([3,3,32,64])
    bias = bias_variable([32])
    deconv_res4 = tf.nn.relu(deconv2d(deconv_res3, filter, conv1_output.shape) + bias)
    # deconv_res4 = tf.layers.batch_normalization(conv_temp,training=True)
    print("deconv2 output : ",deconv_res4.shape) 

with tf.name_scope("deconv1"):
    filter = weight_variable([3,3,3,32])
    bias = bias_variable([3])
    deconv_res5 = tf.nn.relu(deconv2d(deconv_res4, filter, input_shape4d) + bias)
    # deconv_res5 = tf.layers.batch_normalization(conv_temp,training=True)
    print("deconv1 output : ",deconv_res5.shape)
    nn_res = tf.reduce_mean(deconv_res5, 3)
    print("output shape : ",nn_res.shape)

global_step = tf.Variable(0, trainable=False)

one = tf.constant(0)
updata = tf.assign(global_step, one)

mse = tf.losses.mean_squared_error(nn_res, y) 

learning_rate = tf.train.exponential_decay(base_learning_rate, global_step, examples_num / batch_size, decay_rate, staircase=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(mse,global_step=global_step) #使用adam优化器来以0.0001的学习率来进行微调

saver=tf.train.Saver()




def get_next_batch(batchSize,stepNum, shrink,data_name_list):

    batch_num = math.floor(data_name_list.shape[0]/batchSize) #计算batch数 一次step调用一个batch
    curent_batch_index = stepNum % batch_num
    batch_data = data_name_list[curent_batch_index*batchSize:(curent_batch_index + 1)*batchSize].reset_index(drop=True)  #获取batch data
    
    data_name_list = batch_data.str.replace('Camera_5','Camera_5.jpg',n=1)
    label_name_list = batch_data.str.replace('Camera_5','Camera_5.png',n=1)
    
    for count in range(batchSize):
        data_file_path = data_record_dir + data_name_list[count]
        label_file_path = label_record_dir + label_name_list[count]

        if count == 0:
            img = cv2.imread(data_file_path)
            label = cv2.imread(label_file_path)
            
            img=cv2.resize(img,(int(img.shape[1]*shrink),int(img.shape[0]*shrink)),interpolation=cv2.INTER_CUBIC)
            label=cv2.resize(label,(int(label.shape[1]*shrink),int(label.shape[0]*shrink)),interpolation=cv2.INTER_CUBIC)
            
            img = img[np.newaxis,:]
            label = label[np.newaxis,:]
        else :
            temp_img = cv2.imread(data_file_path)
            temp_label = cv2.imread(label_file_path)
            temp_img=cv2.resize(temp_img,(int(temp_img.shape[1]*shrink),int(temp_img.shape[0]*shrink)),interpolation=cv2.INTER_CUBIC)
            temp_label=cv2.resize(temp_label,(int(temp_label.shape[1]*shrink),int(temp_label.shape[0]*shrink)),interpolation=cv2.INTER_CUBIC)
            
            temp_img = temp_img[np.newaxis,:]
            temp_label = temp_label[np.newaxis,:]
            
            img = np.concatenate((img,temp_img),axis=0)
            label = np.concatenate((label,temp_label),axis=0)
    return (img,label)
    
    
with tf.Session() as sess:
    # init = tf.initialize_all_variables()
    # sess.run(init)
    saver.restore(sess, "./module_backup/test")
    # sess.run(updata)
    writer = tf.summary.FileWriter("logs/", sess.graph)
    
    data_record_dir = "../../dataset/img_and_depth/images/Camera 5/"
    label_record_dir = "../../dataset/img_and_depth/depth/Camera 5/"

    data_file_list = pd.Series(os.listdir(data_record_dir))
    data_name_list = data_file_list.str.split('.').str[0].sample(frac=1.0).sample(frac=1.0).sample(frac=1.0).reset_index(drop=True)  #获取到了打乱了的文件名 label和data的名一样 扩展名不一样

    for step in range(step_num):
        batch = get_next_batch(batch_size,step_num,0.1,data_name_list)
        
        _, loss, res, lr = sess.run((train_step, mse, nn_res, learning_rate), feed_dict={x:batch[0], y:batch[1][:,:,:,0]})
        if step % 5 == 0:
            print("writing ...")
            saver.save(sess, "module/test")
            print("write succes")
            
        res[res<0] = 0
        res = res/res.max()*255
        cv2.imwrite("test2.jpg",res[0])

        print("step : ", step, "   loss : ", loss, "     learning rate : ",lr)
        
        
        
        
        
        
        
        
        
        
        
        
