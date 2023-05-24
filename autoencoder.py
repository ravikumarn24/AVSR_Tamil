import speechpy
import numpy as np
import sklearn
import tensorflow as tf
class autoencoder:

    def __init__(self,visible,hidden):
        self.visible=visible
        self.outputnodes=visible
        self.hidden=hidden
        self.iplist=[]
        self.oplist=[]
        self.batch_size = 5
        #self.ip = tf.placeholder('float', [None,self.visible])
        self.ip=tf.placeholder('float')
        self.op = tf.placeholder('float')
        self.prediction = self.neural_network_model(self.ip)
        self.cost =(tf.losses.mean_squared_error(predictions=self.prediction, labels=self.op))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

    def neural_network_model(self,data):
        self.hidden_layer=[]
        self.layer_ops=[]
        for i in range(len(self.hidden)):
         if(i==0):
            h_i = {'weights': tf.Variable(tf.random_normal([self.visible, self.hidden[i]])),
                              'biases': tf.Variable(tf.random_normal([self.hidden[i]]))}
            self.hidden_layer.append(h_i.copy())
            l_i = tf.add(tf.matmul(data, self.hidden_layer[i]['weights']), self.hidden_layer[i]['biases'])
            l_i = tf.nn.sigmoid(l_i)
            self.layer_ops.append(l_i)
         else:
             h_i = {'weights': tf.Variable(tf.random_normal([self.hidden[i-1], self.hidden[i]])),
                    'biases': tf.Variable(tf.random_normal([self.hidden[i]]))}
             self.hidden_layer.append(h_i.copy())
             l_i = tf.add(tf.matmul(self.layer_ops[i-1], self.hidden_layer[i]['weights']), self.hidden_layer[i]['biases'])
             l_i = tf.nn.sigmoid(l_i)
             self.layer_ops.append(l_i)


        self.output_layer = {'weights': tf.Variable(tf.random_normal([self.hidden[len(self.hidden) - 1], self.outputnodes])),
                             'biases': tf.Variable(tf.random_normal([self.outputnodes]))}

        self.output = tf.matmul(self.layer_ops[len(self.layer_ops)-1], self.output_layer['weights']) + self.output_layer['biases']

        return self.output

    def setAudio(self,audio):
        self.frame_audio=audio

    def generateoriginalfeature(self):
        mfcc = speechpy.feature.mfcc(self.frame_audio, 16000, 0.003,0.003)
        mfcc_array = mfcc.flatten()
        mfcc_d1 = speechpy.processing.derivative_extraction(mfcc, 1)
        mfcc_d1_array = mfcc_d1.flatten()
        mfcc_d2 = speechpy.processing.derivative_extraction(mfcc, 2)
        mfcc_d2_array = mfcc_d2.flatten()
        self.originalfeature = np.concatenate((mfcc_array, mfcc_d1_array, mfcc_d2_array))


    def generateFeatures(self,snr=20):
        #mfcc = speechpy.feature.mfcc(self.frame_audio, 16000, 0.015,0.0018)
        mfcc=speechpy.feature.mfcc(self.frame_audio,16000,0.003,0.003)
        mfcc_array = mfcc.flatten()
        #print(len(mfcc_array))
        mfcc_d1 = speechpy.processing.derivative_extraction(mfcc, 1)
        mfcc_d1_array = mfcc_d1.flatten()
        mfcc_d2 = speechpy.processing.derivative_extraction(mfcc, 2)
        mfcc_d2_array = mfcc_d2.flatten()
        self.originalfeature = np.concatenate((mfcc_array, mfcc_d1_array, mfcc_d2_array))
        samples=len(self.frame_audio)
        signal=np.dot(self.frame_audio,self.frame_audio)
        signal_power=np.sum(signal,0)
        noise=np.random.normal(size=samples)
        noise=noise.flatten()
        noise_mag=np.dot(noise,noise)
        noise_power=np.sum(noise_mag,0)
        k=(signal_power/noise_power)*np.power(10,-1*(snr/10))
        noise=int(np.sqrt(k))*noise
        self.corruptedaudio=self.frame_audio+noise.astype(int)
        self.corruptedaudio=np.maximum(self.corruptedaudio,0)
        self.corruptedaudio=np.minimum(self.corruptedaudio,255)
        mfcc = speechpy.feature.mfcc(self.corruptedaudio, 16000, 0.003,0.003)
        mfcc_array = mfcc.flatten()
        mfcc_d1 = speechpy.processing.derivative_extraction(mfcc, 1)
        mfcc_d1_array = mfcc_d1.flatten()
        mfcc_d2 = speechpy.processing.derivative_extraction(mfcc, 2)
        mfcc_d2_array = mfcc_d2.flatten()
        self.corruptedfeature = np.concatenate((mfcc_array, mfcc_d1_array, mfcc_d2_array))
        #print(np.sqrt(np.sum(np.dot(self.corruptedaudio-self.frame_audio,self.corruptedaudio-self.frame_audio))))


    def train_neural_network(self,sess,total_frames):

        hm_epochs = 50
        sess.run(tf.global_variables_initializer())
        prev_epoch=0
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(len(total_frames)):
                for j in range(int(len(total_frames[i]) / self.batch_size)):
                    epoch_x, epoch_y = self.getbatch(total_frames,i,j)
                    with tf.device('/gpu:0'):
                        _, c = sess.run([self.optimizer, self.cost], feed_dict={self.ip: epoch_x, self.op: epoch_y},run_metadata=None)
                    epoch_loss += c
                epoch_loss/=int(len(total_frames) / self.batch_size)
            if(epoch_loss<=0.25):
                    print("Loss is less than threshold(0.25) stopping training ")
                    break
                #print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        if(len(self.iplist)==self.batch_size):
                self.iplist.clear()
                self.oplist.clear()
        if(len(self.iplist)!=self.batch_size):
                self.iplist.append(self.corruptedfeature)
                self.oplist.append(self.originalfeature)
                print(len(self.iplist))
            #correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.originalfeature, 1))

            #accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    def predict_neural_network(self,sess,total_frames):
        tmp=total_frames
        self.iplist.clear()
        for i in range(len(tmp)):
            if(i>=int(len(tmp)/5)*5-1):break
            self.frame_audio = tmp[i]
            #print(len(tmp[i]))
            self.generateoriginalfeature()
            if(len(self.originalfeature)>0):
                self.iplist.append(self.originalfeature)
        xval=self.iplist.copy()
        yval=self.iplist.copy()
        with tf.device('/gpu:0'):
            output,c=sess.run([self.prediction,self.cost],feed_dict={self.ip: xval , self.op: yval})
        print(" audio cost : "+str(c))
        return output

    def getbatch(self,total_frames,i,j):
        tmp=total_frames[i][j*self.batch_size:(j+1)*self.batch_size]

        self.iplist.clear()
        self.oplist.clear()
        for i in range(len(tmp)):
            self.frame_audio=tmp[i]
            self.generateFeatures(30)
            self.iplist.append(self.corruptedfeature)
            self.oplist.append(self.originalfeature)
        return self.iplist,self.oplist




