################################
# Face detection Trainer       #
################################

# import libraries
import warnings
warnings.filterwarnings("ignore")
import datetime
import time
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed, asarray, load, expand_dims
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle



class FaceTrainer:

    def __init__(self):
        self.dataset_train = "D:/PYTHON_CODE/Face_Recognition/faces_dataset/train/"
        self.dataset_val = "D:/PYTHON_CODE/Face_Recognition/faces_dataset/val/"
        self.faces_npz = "D:/PYTHON_CODE/Face_Recognition/faces_dataset.npz"
        self.keras_facenet = "D:/PYTHON_CODE/Face_Recognition/facenet_keras.h5"
        self.faces_embeddings = "D:/PYTHON_CODE/Face_Recognition/faces_dataset_embeddings.npz"
        self.svm_classifier = "D:/PYTHON_CODE/Face_Recognition/SVM_classifier.sav"
        return

    def load_dataset(self, directory):
        """Load a dataset that contains one subdir for each class that in turn contains images"""
        X = []
        y = []
        # enumerate all folders named with class labels
        for subdir in listdir(directory):
            path = directory + subdir + '/'
            # skip any files that might be in the dir
            if not isdir(path):
                continue
            # load all faces in the subdirectory
            faces = self.load_faces(path)
            # create labels
            labels = [subdir for _ in range(len(faces))]
            print("loaded {} examples for class: {}".format(len(faces), subdir))
            X.extend(faces)
            y.extend(labels)
        return asarray(X), asarray(y)

    def load_faces(self, directory):
        """Load images and extract faces for all images in a directory"""
        faces = []
        # enumerate files
        for filename in listdir(directory):
            path = directory + filename
            # get face
            face = self.extract_face(path)
            faces.append(face)
        return faces

    def extract_face(self, filename, required_size=(160, 160)):
        """Extract a single face from a given photograph"""
        # load image from file
        image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = asarray(image)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array

    def create_faces_npz(self):
        """Method Creates npz file for all the faces in train_dir, val_dir"""
        # Load the training data set
        trainX, trainy = self.load_dataset(self.dataset_train)
        print("Training data set loaded")
        # load test dataset
        testX, testy = self.load_dataset(self.dataset_val)
        print("Testing data set loaded")
        # save arrays to one file in compressed format
        savez_compressed(self.faces_npz, trainX, trainy, testX, testy)
        return

    def create_faces_embedding_npz(self):
        """Create npz file for all the face embeddings in train_dir, val_dir"""
        data = load(self.faces_npz)
        trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
        # load the facenet model
        model = load_model(self.keras_facenet)
        print('Keras Facenet Model Loaded')
        # convert each face in the train set to an embedding
        newTrainX = list()
        for face_pixels in trainX:
            embedding = self.get_embedding(model, face_pixels)
            newTrainX.append(embedding)
        newTrainX = asarray(newTrainX)
        # convert each face in the test set to an embedding
        newTestX = list()
        for face_pixels in testX:
            embedding = self.get_embedding(model, face_pixels)
            newTestX.append(embedding)
        newTestX = asarray(newTestX)
        # save arrays to one file in compressed format
        savez_compressed(self.faces_embeddings, newTrainX, trainy, newTestX, testy)
        return

    def get_embedding(self, model, face_pixels):
        """Calculate a face embedding for each face in the dataset using facenet
           Get the face embedding for one face"""
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        return yhat[0]

    def classifier(self):
        """Create a Classifier for the Faces Dataset"""
        # load dataset
        data = load(self.faces_embeddings)
        trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
        # normalize input vectors
        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)
        testX = in_encoder.transform(testX)
        # label encode targets
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        trainy = out_encoder.transform(trainy)
        testy = out_encoder.transform(testy)
        # fit model
        model = SVC(kernel='linear', probability=True)
        model.fit(trainX, trainy)
        # save the model to disk
        filename = self.svm_classifier
        pickle.dump(model, open(filename, 'wb'))
        # predict
        yhat_train = model.predict(trainX)
        yhat_test = model.predict(testX)
        # score
        score_train = accuracy_score(trainy, yhat_train)
        score_test = accuracy_score(testy, yhat_test)
        # summarize
        print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))
        return

    def start(self):
        """Method begins the training process"""
        start_time = time.time()
        st = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        print("-----------------------------------------------------------------------------------------------")
        print("Face trainer Initiated at {}".format(st))
        print("-----------------------------------------------------------------------------------------------")
        # Get faces from the images
        self.create_faces_npz()
        # Get embeddings for all the extracted faces
        self.create_faces_embedding_npz()
        # Classify the faces
        self.classifier()
        end_time = time.time()
        et = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        print("-----------------------------------------------------------------------------------------------")
        print("Face trainer Completed at {}".format(et))
        print("Total time Elapsed {} secs".format(round(end_time - start_time), 0))
        print("-----------------------------------------------------------------------------------------------")

        return


if __name__ == "__main__":
    facetrainer = FaceTrainer()
    facetrainer.start()
