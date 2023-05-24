import tensorflow as tf
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import speechpy
import random
import os
import io
import cv2
import wave
import subprocess
from avsrcode.upload import hmm
from avsrcode.upload import cnn
from avsrcode.upload import autoencoder

# face detector using cascade classifier to extract ROI
face_cascade = cv2.CascadeClassifier("C:/Users/Ravikumar Nagarajan/Documents/python/python-3.6.2.amd64/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml")
numbers=["ஒன்று","இரண்டு","மூன்று","நான்கு","ஐந்து","ஆறு","ஏழு","எட்டு","ஒன்பது","பத்து"]
dae_visible=429
example=[80,40,80]
dae_hidden=[300,150,80,40,80,150,300]
#dae_hidden=[400,300,200,100,200,300,400]
sess=tf.Session()
sess.run(tf.global_variables_initializer())
dae=autoencoder.autoencoder(dae_visible,dae_hidden)
CNN=cnn.CNN()
hmm_models=[]
codebook=np.array([])
kmc=KMeans(n_clusters=16)
knn_labels=[]
knn_model=KNeighborsClassifier()
for __ in range(10):
    hmm_models.append(hmm.hmm(3))
phase=0 #0 implies training phase 1 implies testing phase
trainingdir="trimming/train/digits"
testingdir="trimming/test/digits"
vector_list=[]
final_op=[]
total_epoch=2
total_audio_frames=[]
audio_obtained=False
while(phase<2):
    videoclips=[]
    if(phase==0):print("#####TRAINING PHASE####"+str(5-total_epoch+1))
    else : print("####TESTING PHASE#####")
    if(phase==0):digits=os.listdir(trainingdir)
    else : digits=os.listdir(testingdir)
    for d in digits:
        if(phase==0):videoclips+=os.listdir(trainingdir+"/"+d)
        else :videoclips+=os.listdir(testingdir+"/"+d)
    if(total_epoch>1):
        c=list(zip(digits,videoclips))
        np.random.shuffle(c)
        digits,videoclips=zip(*c)
        c.clear()
    for iteratr in range(len(videoclips)):
        if(phase==0):clip =trainingdir+"/"+digits[iteratr]+"/"+ videoclips[iteratr]
        else :clip =testingdir+"/"+digits[iteratr]+"/"+ videoclips[iteratr]
        print(clip)
        #ffmpeg framework for extracting audio from the video
        command = "ffmpeg -y -i "+clip+" -ar 16000 -vn audio.wav"
        subprocess.call(command, shell=True)
        vid=cv2.VideoCapture(clip)
        audio=wave.open("audio.wav")
        total_frames=[]
        tmp_cube=[]
        visual_cube=[]
        frame_cnt=-1
        cnn_labels=[]
        cnn_frame_num=[]
        tmp_num=[]
        while vid.isOpened():
          with tf.device('/cpu:0'):
            ret,frame=vid.read()
            if(ret!=True):break
            frame_cnt+=1
            frame = cv2.resize(frame, (480, 640))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #gray = np.transpose(gray)
            fps = vid.get(cv2.CAP_PROP_FPS)
            #cv2.equalizeHist(gray,gray)
            faces = face_cascade.detectMultiScale(gray,1.05,7)
            roi_gray=gray
            roi_color=frame
            const=100
            #print(frame_cnt)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_color=frame[y:y + h, x:x + w]
            #cv2.imshow("frame", roi_gray)
            mouth=[]
            frame_audio = audio.readframes(533)
            mono=[]
            for i in range(0,len(frame_audio),2):
                mono.append(((frame_audio[i])+(frame_audio[i+1]))/2)
                if(mono[int(i/2)]<0):print("yes")
            frame_audio=mono
            frame_audio = np.asarray(list(frame_audio))
            total_frames.append(frame_audio)
            if(len(faces)>0 and len(roi_gray)>const):
                # lip extraction from the detected face
                x=(int)(0.28*len(roi_gray[0]))
                y=(int)(0.58*len(roi_gray))
                h=(int)(0.35*len(roi_gray))
                w=(int)(0.42*len(roi_gray[0]))
                mouth=roi_gray[y:y+h,x:x+w]
                mouth=cv2.resize(mouth,(70,60),interpolation=cv2.INTER_AREA)
                tmp_cube.append(mouth)
                cnn_labels.append(frame_audio)
                tmp_num.append(frame_cnt)
                if(len(tmp_cube)==5):
                    visual_cube.append(tmp_cube.copy())
                    cnn_frame_num+=tmp_num.copy()
                    tmp_num.clear()
                    tmp_cube.clear()
                #cv2.imshow("mouth",mouth)
                #cv2.waitKey(0)
           # if(len(mouth)!=0):
            #dae.setAudio(frame_audio)
            #dae.generateFeatures(snr=10)
                #print("hi")
        print(len(total_frames))
        if(audio_obtained==False):total_audio_frames.append(total_frames);continue
        nn_output=[]
        with tf.device('/gpu:0'):
            if(total_epoch>0 and iteratr==0):
                #print(len(total_audio_frames[0]))
                dae.train_neural_network(sess,total_audio_frames)
            audiooutput=dae.predict_neural_network(sess,total_frames)
            tmp=[]
            for i in range(len(cnn_labels)):
                dae.frame_audio=cnn_labels[i]
                dae.generateoriginalfeature()
                tmp.append(dae.originalfeature.copy())
            cnnlabels=[]
            for i in range(int(len(tmp)/5)):
                label=[]
                if(i*5+4<len(tmp)):
                    for j in range(len(tmp[i*5])):
                        label.append((tmp[i*5][j]+tmp[i*5+1][j]+tmp[i*5+2][j]+tmp[i*5+3][j]+tmp[i*5+4][j])/5)
                cnnlabels.append(label)
            if(total_epoch>0 and phase==0):CNN.train_network(sess,visual_cube,cnnlabels)
            videooutput=CNN.test_network(sess,visual_cube,cnnlabels)
            itr=0;frcnt=0
            #audiooutput=np.array(tmp)
            #print(len(audiooutput))
            if(total_epoch>1):continue
            for i in range(len(audiooutput)):
                if(i in cnn_frame_num):
                    if(len(nn_output)>0):
                        tmp=(0.64*audiooutput[i]+0.36*videooutput[itr])
                        #nn_output=np.concatenate((nn_output,tmp))
                    else :tmp=(0.64*audiooutput[i]+0.36*videooutput[itr]);#nn_output=np.copy(tmp);
                    if(total_epoch==1):codebook=np.concatenate((codebook,tmp))
                    else : nn_output=np.concatenate((nn_output,np.array(knn_model.predict([tmp.tolist()]))+1))
                    frcnt+=1
                    if(frcnt==5):itr+=1;frcnt=0
                else:
                    #if(len(nn_output)>0):nn_output=np.concatenate((nn_output,audiooutput[i]))
                    #else : nn_output=audiooutput[i]
                    tmp=audiooutput[i]
                    if(total_epoch==1):codebook=np.concatenate((codebook,tmp))
                    else : nn_output=np.concatenate((nn_output,np.array(knn_model.predict([tmp.tolist()]))+1))

            if(total_epoch<=0):
                #print(nn_output)
                arr_size=6-int(len(nn_output)%6)
                nn_output=np.concatenate((nn_output,np.zeros(arr_size)))
                print(len(nn_output))
                vector=np.array(nn_output)
                vector_list.append(vector)
                vector=vector.reshape(3,-1)
                print(vector.shape)
                #file.write(vector)
                #vector+=abs(np.min(vector))+0.00000001
                vector/=vector.sum(axis=0)
            if(total_epoch==0 and phase==0):
                hmm_models[iteratr].fit(vector)
                #for tt in range(iteratr+1):
                    #print(hmm_models[tt].transform(vector))
            elif(phase==1):
                mxval=-1000000;mxindex=-1
                index_cnt=0
                for tmp in hmm_models:
                    vl=tmp.transform(vector)
                    print(vl)
                    if(mxval<vl):
                        mxval=vl
                        mxindex=index_cnt
                    index_cnt += 1
                print("The predicted val is "+numbers[mxindex])
                final_op.append(numbers[mxindex])
    audio_obtained=True
    total_epoch-=1
    if(total_epoch==0):
        kmc=kmc.fit(codebook.reshape(-1,429))
        knn_labels=kmc.predict(codebook.reshape(-1,429))
        print(knn_labels)
        knn_model.fit(codebook.reshape(-1,429),np.array(knn_labels))
    if(total_epoch<0):phase+=1
    if(phase==2):
        for i in range(len(videoclips)):
            print("Video Clip : "+videoclips[i] +" Predicted word : " + final_op[i]+"  Actual Word : "+numbers[i])



vid.release()
cv2.destroyAllWindows()