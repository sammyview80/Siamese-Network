import os 
import numpy as np
import tensorflow as tf 
import cv2 as cv
import matplotlib.pyplot as plt 
from data import LoadData
from siamese import Models
from tensorflow.keras.models import model_from_json
from mtcnn import MTCNN




class Main():
    def __init__(self):
        pass 

    def load_data(self):
        print('Loading Dataset...')
        self.x_right, self.x_left, self.y_train = LoadData().load()


    def create_model(self):
        print('Creating Model...')
        model = Models()
        model.create_model()
        model.summary()
        model.compile()
        model.fit(self.x_left, self.x_right, self.y_train, epochs=5)
        model.save()
        
    def run(self, load_data=True):
        self.load_data()
        self.create_model()

    def unique_output(self):
        y_train = np.load('Savedtraintestdata/y_train.npy')
        # print(y_train.shape)
        return np.unique(y_train)

    def load_model(self):
        return tf.keras.models.load_model('save/super.h5') 



class Preprocessing():
    def __init__(self):
        self.DBfile = os.listdir('database')
        self.path = 'database/'
        self.detector = MTCNN()
        self.files = []

    def get_face(self, image):
        
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        results = self.detector.detect_faces(image)

        # print(results[0])
        x1, y1, width, height = results[0]['box']

        x1, y1 = abs(x1), abs(y1)

        face = image[y1:y1+height, x1:x1+width]

        # plt.imshow(face, plt.cm.binary)
        # plt.show()

        # plt.imshow(face, plt.cm.binary)
        # plt.show()

        # print(face.shape)
            
    

        return face         


    def setup(self):
        m = Main()
        # m.load_data()
        self.model = m.load_model()
        self.unique_output = m.unique_output() 
        
    def read_file(self):
        for i in range(0, len(self.DBfile)):
            DBimage = cv.imread(f'{self.path}{self.DBfile[i]}', 1)
            DBimage = self.get_face(DBimage)
            DBimage = self.preprocess(DBimage)
            self.files.append(DBimage)
        # return self.files 

    def resize(self, image):
        return cv.resize(image, (35, 35))


    def preprocess(self, image):
        image = self.resize(image)
        image = image.reshape(1, 35, 35, 3)
        image = image/255
        # print(image.shape)
        return image 

    def compare(self, imageArray):
        # print(imageArray.shape)
        imageArray = self.preprocess(imageArray)
        
        for i in range(0, len(self.DBfile)):
            prediction = self.model.predict([self.files[i], imageArray])

            # print(prediction)
            prediction = self.unique_output[np.argmax(prediction)]

            if prediction == 1:
                # print(self.DBfile[i])
                return self.DBfile[i].split('.')[0]

if __name__ == "__main__":
    # # m = Main()
    m = Main().run()
    # # model = m.load_model()
    # # model.summary()
    # # print(type(model))
    # pass