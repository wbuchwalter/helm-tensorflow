import tensorflow as tf
import numpy as np
from skimage.data import astronaut
from scipy.misc import imresize, imsave, imread

img = imread('./doge.jpeg')
img = imresize(img, (100, 100))
save_dir = 'output'

def distance(p1, p2):
    return tf.abs(p1 - p2)

def negative_color_distance(p1, p2):
    n = [255, 255, 255]
    target = (p1 - n) * -1
    return tf.abs(target - p2)

def linear_layer(X, layer_size, layer_name):
    with tf.variable_scope(layer_name):
        W = tf.Variable(tf.random_uniform([X.get_shape().as_list()[1], layer_size], dtype=tf.float32))
        b = tf.Variable(tf.zeros([layer_size]))
        return tf.nn.relu(tf.matmul(X, W) + b)

def main():
    X = tf.placeholder(dtype=tf.float32, shape=(None, 2)) 
    y = tf.placeholder(dtype=tf.float32, shape=(None, 3))
    input = X
    for layer_id in range(4):
        h = linear_layer(input, 64, 'layer{}'.format(layer_id))
        input = h
    y_pred = linear_layer(input, 3, 'output')

    #loss will be distance between predicted and true RGB
    loss = tf.reduce_sum(distance(y_pred, y))
    train_op = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)

    xs = []
    ys = []
    for row_i in range(img.shape[0]):
        for col_i in range(img.shape[1]):
            xs.append([row_i, col_i])
            ys.append(img[row_i, col_i])

    xs = (xs - np.mean(xs)) / np.std(xs)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(200000):
            sess.run(train_op, feed_dict={X:xs, y:ys})
            training_cost = sess.run(loss, feed_dict={X: xs, y: ys})
            if i % 100 == 0:
                print(i, training_cost)
            if i % 2500 == 0:
                output = sess.run(y_pred, feed_dict={X: xs})
                res_img = np.clip(output.reshape(img.shape), 0, 255).astype(np.uint8)
                imsave('{}/step_{}.jpg'.format(save_dir, i), res_img)
        output = sess.run(y_pred, feed_dict={X: xs})
        print(output)
if __name__ == "__main__":
    main()