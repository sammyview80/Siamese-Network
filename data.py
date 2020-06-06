import os 
import time
import random 
import numpy as np 
import tensorflow as tf 
import cv2 as cv 
import matplotlib.pyplot as plt 
from mtcnn import MTCNN


class LoadData():
    def __init__(self):
        # Loading the data 
        self.detectot = MTCNN()

    def create_data(self):
        fotos = os.listdir('foto/')
        print(fotos)
        for foto in fotos:
            files = os.listdir('data/')
            ids = foto.split('-')
            if len(ids) == 2:
                id = ids[0]
                if str(id) in files:
                    os.replace(f'foto/{foto}', f'data/{id}/{foto}')
                else:
                    os.mkdir(f'data/{id}')
                

    def get_dataset(self):
        """"
        This create a array of image (0-234.jpg, 0-224.jpg) and label of 1 ie same
        """
        files = os.listdir('data/')
        files.sort()
        
        path = 'data/'
        pre_data1 = []
        labels1 = []
        for i in files:
            dir_files = os.listdir(f'data/{i}')
            path = f'data/{i}'
            for j in range(0, len(dir_files)-1):
                image1 = cv.imread(os.path.join(path, dir_files[j]), 1)
                image2 = cv.imread(os.path.join(path, dir_files[j+1]), 1)
                # print(image1.shape)
                image1 = self.get_face(image1)
                image2 = self.get_face(image2)
                pre_data1.append(np.array((image1, image2)))
                labels1.append(1)
                # pre_data1.append(np.array((image1, np.flip(image2))))
                # labels1.append(1)
                # pre_data1.append(np.array((np.rot90(image1), image2)))
                # labels1.append(1)
                # pre_data1.append(np.array((np.flip(image1), np.rot90(image2))))
                # labels1.append(1)


        pre_data2 = []
        labels2 = []
        for i in range(0, len(files)-1):
            path = f'data/{files[i]}'
            dir_files = os.listdir(f'data/{files[i]}')
            dir_files_following = os.listdir(f'data/{files[i+1]}')
            for j in range(0, min(len(dir_files)-1, len(dir_files_following)-1)):
                if dir_files_following[j]:
                    image1 = cv.imread(f'{path}/{dir_files[j]}', 1)
                    image2 = cv.imread(f'data/{files[i+1]}/{dir_files_following[j]}', 1)
                    image1 = self.get_face(image1)
                    image2 = self.get_face(image2)
                    pre_data2.append(np.array((image1, image2)))
                    labels2.append(0)
                    # pre_data2.append(np.array((image1, np.flip(image2))))
                    # labels2.append(0)

        dataset1 = self.merge_data(pre_data1, labels1)
        dataset2 = self.merge_data(pre_data2, labels2)
        np.random.shuffle(dataset1)
        np.random.shuffle(dataset2)

        dataset = dataset1 + dataset2 

        dataset = np.array(dataset)
        # dataset = np.random.shuffle(dataset)
        np.random.shuffle(dataset)
        
        # print(dataset[0].shape)

        return dataset 

    def merge_data(self, pre_data, label):
        dataset = []
        for i in range(0, len(pre_data)):
            dataset.append((pre_data[i], label[i]))

        return dataset 

    def split_train_test(self, dataset):
        x_train = []
        y_train = []
        for i in dataset:
            x_train.append(i[0])
            y_train.append(i[1])

        return np.array(x_train), np.array(y_train)

    def split_left_right_input(self, x_train):
        left_input = []
        right_input = []

        for i in range(0, len(x_train)):
            left_input.append(x_train[i][0])
            right_input.append(x_train[i][1])
            
        return np.array(left_input), np.array(right_input)

    def normalize(self, x, y):
        return x/255, y/255

    def load(self):
        # self.create_files_folder()
        dataset = self.get_dataset()
        x_train, y_train = self.split_train_test(dataset)
        x_train_left, x_train_right = self.split_left_right_input(x_train)

        x_train_right, x_train_left = self.normalize(x_train_right, x_train_left) 
        # x_train_right = x_train_right[:, :, :, tf.newaxis]
        # x_train_left = x_train_left[:, :, :, tf.newaxis]

        np.save('Savedtraintestdata/left.npy', x_train_left) # save


        np.save('Savedtraintestdata/right.npy', x_train_right) # save
        np.save('Savedtraintestdata/y_train.npy', y_train) # save


        return x_train_left, x_train_right, y_train


        

    def get_face(self, image):
        
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        results = self.detectot.detect_faces(image)

        try:
            # print(results[0])
            x1, y1, width, height = results[0]['box']

            x1, y1 = abs(x1), abs(y1)

            face = image[y1:y1+height, x1:x1+width]

            # plt.imshow(face, plt.cm.binary)
            # plt.show()

            # print(face.shape)
            
        
        except IndexError:
            face = image 

        face = cv.resize(face, (35, 35))

        return face         

        # new_num_arr = np.load('data.npy') # load


# class TripletLossData():
#     def __init__(self):
#         self.main_folder_path = 'data'
#         self.detectot = MTCNN()

#     def get_dataset(self, read_asarray = False):
#         main_folder = os.listdir(self.main_folder_path)
#         for i in range(0, len(main_folder)-1):
#             # Getinto first folder 
#             folder1 = os.listdir('{}/{}'.format(self.main_folder_path, main_folder[i]))
#             folder2 = os.listdir('{}/{}'.format(self.main_folder_path, main_folder[i+1]))
#             folder1_path = '{}/{}'.format(self.main_folder_path, main_folder[i])
#             folder2_path = '{}/{}'.format(self.main_folder_path, main_folder[i+1])
#             for j in range(0, min(len(folder1), len(folder2))-1):
#                 anchor = '{}/{}'.format(folder1_path, folder1[j])
#                 positive = '{}/{}'.format(folder1_path, folder1[j+1])
#                 negative = '{}/{}'.format(folder2_path, folder2[j])
#                 if read_asarray:
#                     #Getting the face with filepath
#                     anchor_array = self.get_face(anchor)
#                     positive_array = self.get_face(positive)
#                     negative_array = self.get_face(negative)
                    
#                     # print(f'[{anchor_array.shape, positive_array.shape, negative_array.shape}]')



                
                # """"
                # print(f'[{anchor, positive, negative}]')
                # results:
                # [('29-p290.jpg', '29-p294.jpg', 'Queen_Elizabeth_II_0009.jpg')]
                # [('29-p294.jpg', '29-p296.jpg', 'Queen_Elizabeth_II_0011.jpg')]
                # [('29-p296.jpg', '29-p291.jpg', 'Queen_Elizabeth_II_0004.jpg')]
                # [('29-p291.jpg', '29-p297.jpg', 'Queen_Elizabeth_II_0001.jpg')]
                # [('29-p297.jpg', '29-p298.jpg', 'Queen_Elizabeth_II_0006.jpg')]
                # [('29-p298.jpg', '29-p293.jpg', 'Queen_Elizabeth_II_0002.jpg')]
                # [('29-p293.jpg', '29-p295.jpg', 'Queen_Elizabeth_II_0003.jpg')]
                
                # """
        # folder = os.listdir(self.data_path)
        # next_files = os.listdir(self.file_path.format(files[i+1]))
        # for files in range(0, len(folder)):
        #     folder_inside = os.listdir(self.file_path.format(files[i]))
        #     image1 = folder_inside[i]
        #     image2 = folder_inside[i+1]
        #     image2 = 
    #         # Reading image with cv 
    # def get_face(self, imagePath):

    #     image = cv.imread(imagePath, 1)
    
    #     image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    #     results = self.detectot.detect_faces(image)

    #     try:
    #         # print(results[0])
    #         x1, y1, width, height = results[0]['box']

    #         x1, y1 = abs(x1), abs(y1)

    #         face = image[y1:y1+height, x1:x1+width]

    #         # plt.imshow(face, plt.cm.binary)
    #         # plt.show()

    #         # print(face.shape)
            
        
    #     except IndexError:
    #         face = image 

    #     face = cv.resize(face, (35, 35))

    #     return face         

if __name__ == "__main__":
    # LoadData().create_data()
    start = time.time()
    x1, x2, y = LoadData().load()
    end = time.time()
    print(f'Total_Time: {end - start}')
    print(x1.shape, x2.shape, y.shape)
    # start = time.time()
    # t = TripletLossData()
    # t.get_dataset(read_asarray=True)
    # end = time.time()
    # print(f'Total_Time: {end - start}')
    

  