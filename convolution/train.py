import tensorflow as tf
import input_data
import cv2
import numpy as np
import math
from scipy import ndimage
import os





def train(iterations):

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x_input = tf.placeholder("float", [None, 784], name="x_input")
    y_target = tf.placeholder("float", [None, 10], name="y_target")
    x_input_image=tf.reshape(x_input,[-1,28,28,1])
    init_random_dist_1 = tf.truncated_normal([6,6,1,32], stddev=0.1)
    W_layer_1=tf.get_variable("W_layer_1",shape=[6,6,1,32],initializer =tf.contrib.layers.xavier_initializer())
    b_layer_1=tf.get_variable("b_layer_1",shape=[32],initializer =tf.contrib.layers.xavier_initializer())
    convolution_layaer_1=tf.nn.relu(tf.nn.conv2d(x_input_image,W_layer_1,strides=[1,1,1,1],padding="SAME")+b_layer_1)
    convolution_pooling_1=tf.nn.max_pool(convolution_layaer_1,ksize=[1,2,2,1],
    strides=[1,2,2,1],padding="SAME")

    init_random_dist_2 = tf.truncated_normal([6,6,32,64], stddev=0.1)
    W_layer_2=tf.get_variable("W_layer_2",shape=[6,6,32,64],initializer =tf.contrib.layers.xavier_initializer())
    b_layer_2=tf.get_variable("b_layer_2",shape=[64],initializer =tf.contrib.layers.xavier_initializer())
    convolution_layaer_2=tf.nn.relu(tf.nn.conv2d(convolution_pooling_1,W_layer_2,strides=[1,1,1,1],padding="SAME")+b_layer_2)
    convolution_pooling_2=tf.nn.max_pool(convolution_layaer_2,ksize=[1,2,2,1],
    strides=[1,2,2,1],padding="SAME")

    covolution_layer_2_flat=tf.reshape(convolution_pooling_2,[-1,7*7*64])

    

    init_random_dist_fully_layer = tf.truncated_normal([int(covolution_layer_2_flat.get_shape()[1]),1024], stddev=0.1)
    W_fully_layer=tf.get_variable("W_fully_layer",shape=[int(covolution_layer_2_flat.get_shape()[1]),1024],initializer =tf.contrib.layers.xavier_initializer())
    b_fully_layer=tf.get_variable("b_fully_layer",shape=[1024],initializer =tf.contrib.layers.xavier_initializer())
    fully_connected_layer = tf.nn.relu(tf.matmul(covolution_layer_2_flat,W_fully_layer)+b_fully_layer)
    hold_prob = tf.placeholder(tf.float32)
    fully_connected_layer_dropout = tf.nn.dropout(fully_connected_layer,keep_prob=hold_prob)

    init_random_dist_final = tf.truncated_normal([int(fully_connected_layer_dropout.get_shape()[1]),10], stddev=0.1)
    W_final=tf.get_variable("W_final",shape=[int(fully_connected_layer_dropout.get_shape()[1]),10],initializer =tf.contrib.layers.xavier_initializer())
    b_final=tf.get_variable("b_final",shape=[10],initializer =tf.contrib.layers.xavier_initializer())
    y_prediction=tf.nn.relu(tf.matmul(fully_connected_layer_dropout,W_final)+b_final)

    #Loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target,logits=y_prediction))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train = optimizer.minimize(cross_entropy)

    

    saver = tf.train.Saver(save_relative_paths=True)
    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
    session = tf.Session(config=config)
    with session:
        session.run(tf.global_variables_initializer())

        for i in range(iterations):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            session.run(train,feed_dict={x_input:batch_xs,y_target:batch_ys,hold_prob:0.2})
            correct_prediction = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(y_target, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            if (i % 100) == 0:
                print(session.run(accuracy, feed_dict={
                    x_input: mnist.test.images, y_target: mnist.test.labels,hold_prob:1.0}))
        saver.save(session, "./model")

#train(1000)


def make_prediction(image_1):
    print("making a prediction....")
    tf.reset_default_graph()
    x_input = tf.placeholder("float", [None, 784], name="x_input")
    y_target = tf.placeholder("float", [None, 10], name="y_target")
    x_input_image=tf.reshape(x_input,[-1,28,28,1])
    init_random_dist_1 = tf.truncated_normal([6,6,1,32], stddev=0.1)
    W_layer_1=tf.get_variable("W_layer_1",shape=[6,6,1,32],initializer =tf.contrib.layers.xavier_initializer())
    b_layer_1=tf.get_variable("b_layer_1",shape=[32],initializer =tf.contrib.layers.xavier_initializer())
    convolution_layaer_1=tf.nn.relu(tf.nn.conv2d(x_input_image,W_layer_1,strides=[1,1,1,1],padding="SAME")+b_layer_1)
    convolution_pooling_1=tf.nn.max_pool(convolution_layaer_1,ksize=[1,2,2,1],
    strides=[1,2,2,1],padding="SAME")


    init_random_dist_2 = tf.truncated_normal([6,6,32,64], stddev=0.1)
    W_layer_2=tf.get_variable("W_layer_2",shape=[6,6,32,64],initializer =tf.contrib.layers.xavier_initializer())
    b_layer_2=tf.get_variable("b_layer_2",shape=[64],initializer =tf.contrib.layers.xavier_initializer())
    convolution_layaer_2=tf.nn.relu(tf.nn.conv2d(convolution_pooling_1,W_layer_2,strides=[1,1,1,1],padding="SAME")+b_layer_2)
    convolution_pooling_2=tf.nn.max_pool(convolution_layaer_2,ksize=[1,2,2,1],
    strides=[1,2,2,1],padding="SAME")

    covolution_layer_2_flat=tf.reshape(convolution_pooling_2,[-1,7*7*64])

    init_random_dist_fully_layer = tf.truncated_normal([int(covolution_layer_2_flat.get_shape()[1]),1024], stddev=0.1)
    W_fully_layer=tf.get_variable("W_fully_layer",shape=[int(covolution_layer_2_flat.get_shape()[1]),1024],initializer =tf.contrib.layers.xavier_initializer())
    b_fully_layer=tf.get_variable("b_fully_layer",shape=[1024],initializer =tf.contrib.layers.xavier_initializer())
    fully_connected_layer = tf.nn.relu(tf.matmul(covolution_layer_2_flat,W_fully_layer)+b_fully_layer)
    hold_prob = tf.placeholder(tf.float32)
    fully_connected_layer_dropout = tf.nn.dropout(fully_connected_layer,keep_prob=hold_prob)


    init_random_dist_final = tf.truncated_normal([int(fully_connected_layer_dropout.get_shape()[1]),10], stddev=0.1)
    W_final=tf.get_variable("W_final",shape=[int(fully_connected_layer_dropout.get_shape()[1]),10],initializer =tf.contrib.layers.xavier_initializer())
    b_final=tf.get_variable("b_final",shape=[10],initializer =tf.contrib.layers.xavier_initializer())
    y_prediction=tf.nn.relu(tf.matmul(fully_connected_layer_dropout,W_final)+b_final)

    #Loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target,logits=y_prediction))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train = optimizer.minimize(cross_entropy)

    flatten = image_1.flatten() / 255.0
    images = np.zeros((1, 784))
    correct_val = np.zeros((1, 10))
    images[0] = flatten
    new_saver = tf.train.Saver()
    with tf.Session() as sess:
        
          

        new_saver=tf.train.Saver(tf.all_variables())
        new_saver=tf.train.import_meta_graph('model.meta')
        new_saver.restore(sess,"model")
        prediction = tf.argmax(y_prediction, 1)
        print("Model predicted...", sess.run(prediction,
                                           feed_dict={x_input: images,hold_prob:1.0}))
