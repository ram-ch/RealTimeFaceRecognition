# Real Time Face Recognition   
This project aims at building a realtime face recognition model. I have used 
OpenCV - For real time camera feed from laptop webcam
Multi-task Cascaded Convolutional Networks (MTCNN) - For face extraction from an image
keras facenet model - For creating embeddings of the extracted faces
Support Vector Machines algorithm - For predicting the target variable class

### Files and directory structure   
```bash
RealTimeFaceRecognition
|__FaceDetector.py      
|__FaceTrainer.py     
|__facenet_keras.h5    
|__SVM_classifier.sav    
|__faces_dataset_embeddings.npz   
|__faces_dataset.npz   
|__faces_dataset   
   |__train   
      |__sai_ram    
      |__donald_trump   
      |__narendra_modi   
      |__virat_kohli   
   |__val   
      |__sai_ram   
      |__donald_trump   
      |__narendra_modi   
      |__virat_kohli   
```


### Training Data  
The training data consists of 4 classes (4 persons)
1. sai_ram
2. narendra_modi
3. donald_trump
4. virat_kohli  
I have gathered 15 images per class and have compiled the training data set with a total of 60 images. The directory is considered as the class label name


### FaceTrainer.py   
This file trains a model on the images in faces_dataset

### FaceDetector.py   
This file uses live web cam feed to detect and recognize the faces


MTCNN

Keras facenet pretrained model

Support Vector Machines

