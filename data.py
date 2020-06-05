import os 
import random 
import numpy as np 
import tensorflow as tf 
import cv2 as cv 
import matplotlib.pyplot as plt 



class LoadData():
    def __init__(self):
        # Loading the data 
        pass

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
                image1 = cv.resize(image1, (28, 28))
                image2 = cv.resize(image2, (28, 28))

                pre_data1.append(np.array((image1, image2)))
                labels1.append(1)

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
                    image1 = cv.resize(image1, (28, 28))
                    image2 = cv.resize(image2, (28, 28))

                    pre_data2.append(np.array((image1, image2)))
                    labels2.append(0)

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

        np.save('left.npy', x_train_left) # save


        np.save('Savedtraintestdata/right.npy', x_train_right) # save
        np.save('Savedtraintestdata/y_train.npy', y_train) # save


        return x_train_left, x_train_right, y_train


        

        

        # new_num_arr = np.load('data.npy') # load



if __name__ == "__main__":
    # LoadData().create_data()
    x1, x2, y = LoadData().load()
    print(x1.shape, x2.shape, y.shape)
    

  