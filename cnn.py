import tensorflow as tf
import numpy as np

class CNN:
    def __init__(self):
        self.outputnodes =429
        self.iplist = []
        self.oplist = []
        self.batch_size = 10
        self.ip= tf.placeholder('float')
        self.labels = tf.placeholder('float')
        self.prediction = self.neural_network_model(self.ip)
        self.cost = (tf.losses.mean_squared_error(predictions=self.prediction, labels=self.labels))
        self.diff=tf.losses.absolute_difference(predictions=self.prediction,labels=self.labels)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(self.cost)

    def neural_network_model(self, data):
        self.hidden_layer = []
        self.layer_ops = []
        input_layer = tf.reshape(data, [-1, 5 ,60, 70, 1])

        # Convolutional Layer #1 op size : 3*58*68*16
        conv1 = tf.layers.conv3d(inputs=input_layer,filters=16,kernel_size=[3,3,3],activation=tf.nn.relu)
        # Pooling Layer #1 op size : 3*28*33*16
        pool1 = tf.layers.max_pooling3d(inputs=conv1,pool_size=[1,3,3],strides=[1,2,2])

        # Convolutional Layer #2 op size : 1*26*31*32
        conv2 = tf.layers.conv3d(inputs=pool1, filters=32, kernel_size=[3, 3, 3], activation=tf.nn.relu)
        # Pooling Layer #2 op size : 1*12*15*32
        pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[1,3, 3], strides=[1,2,2])

        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 1*12*15*32])
        self.op = tf.layers.dense(inputs=pool2_flat, units=self.outputnodes,activation=tf.nn.leaky_relu)
        return self.op

    def train_network(self,sess,data,labels):
        epochs=50
        for i in range(epochs):
            epoch_loss=0
            for j in range(len(data)):
                x=data[j]
                y=labels[j]
                with tf.device('/gpu:0'):
                    _,c=sess.run([self.optimizer,self.cost],feed_dict={self.ip:x,self.labels:y})
                    epoch_loss+=c
            epoch_loss/=len(data)
            if(epoch_loss<=0.05):
                print("Loss is less than threshold(0.05) stopping training")
                break
            #print("Epoch "+str(i)+" : loss "+str(epoch_loss))

    def test_network(self,sess,data,labels):
        with tf.device('/gpu:0'):
            predictedval,c=sess.run([self.prediction,self.diff],feed_dict={self.ip:data,self.labels:labels})
            print("Final video cost : "+str(c))
            return predictedval









