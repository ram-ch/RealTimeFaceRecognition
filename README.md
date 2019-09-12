# Real Time Face Recognition   
This project aims at building a realtime face recognition model. I have used       
* OpenCV - For real time camera feed from laptop webcam     
* Multi-task Cascaded Convolutional Networks (MTCNN) - For face extraction from an image     
* keras facenet model - For creating embeddings of the extracted faces       
* Support Vector Machines algorithm - For predicting the target variable class     

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
The training data consists of 4 classes (4 persons). I have gathered 15 images per class and have compiled the training data set with a total of 60 images. The directories is named after the class labels. Apart from training data I have gathered 5 more images for validation during the training of the model. Initially I thought of compiling a data set with thousands of images for each class, but in real world business use cases it is very difficut to gather such a huge number of pictures for each individual. Hence our model should be good enough, even with bare minimum number of observation for each class.
1. sai_ram
2. narendra_modi
3. donald_trump
4. virat_kohli     

### FaceTrainer.py    
This python script builds and trains a model on the images in faces_dataset   
**Training Procedure:**  
* Step 1:     
* Step 2:    

### FaceDetector.py   
This file uses live web cam feed to detect and recognize the faces


MTCNN

Keras facenet pretrained model

Support Vector Machines

