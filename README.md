# Real-Time Face Recognition   
This project aims at building a realtime face recognition model.The purpose is to recognize a person/persons in a natural video. The model takes in a camera feed and returns a video stream with a bounding box and a probability for all the class labels. To achieve this functionality I have used:      
* OpenCV - For real-time camera feed from a laptop webcam     
* Multi-task Cascaded Convolutional Networks (MTCNN) - For face extraction from an image     
* Keras face net model - For creating embeddings of the extracted faces       
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
The training data consists of 4 classes (4 persons). I have gathered 15 images per class and have compiled the training data set with a total of 60 images. The directories are named after the class labels. Apart from training data I have gathered 5 more images for validation during the training of the model. Initially, I thought of compiling a data set with thousands of images for each class, but in real-world business use cases, it is very difficult to gather such a huge number of pictures for each individual. Hence our model should be good enough, even with bare minimum number of observation for each class.
1. sai_ram
2. narendra_modi
3. donald_trump
4. virat_kohli     

### FaceTrainer.py    
This python script builds and trains a model on the images in faces_dataset      
   
**Training Procedure:**  
* Step 1: Face Detection from natural images        
There are 2 major issues to be addressed in this project. One is detecting a face in a natural image/video (face detection) and the second part is recognizing the class labels of the detected faces. I have used Multi-task Cascaded Convolutional Networks (MTCNN) to detect a face in an image. The faces detected by MTCNN are used for further processing in the pipeline.      
* Step 2: Creating .npz file for the extracted faces   
The extracted faces are converted and saved as numpy array zip files in python
* Step 3: Creating embeddings(high-level features) for the extracted face    
* Step 4: Training a Support Vector Machine(SVM) on the embeddings   
* Step 5: Saving the trained SVM model for future inferences   

### FaceDetector.py   
This file uses live web cam feed to detect and recognize the faces


MTCNN

Keras facenet pretrained model

Support Vector Machines

