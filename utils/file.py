import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt 
faces = np.load('np/olivetti_faces.npy')
labels = np.load('np/olivetti_faces_target.npy')

reshape_faces = []
for i in faces:
    a = np.multiply(i, 255)
    # a = a.reshape(64, 64, 1)
    reshape_faces.append(a)

r = np.array(reshape_faces)

for i in range(0, len(faces)):
    cv.imwrite(f'../foto/{labels[i]}-p{i}.jpg', r[i])


# plt.imshow(r[0])

# plt.show()