from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
from keras_facenet import FaceNet
import numpy as np
import os

class FaceClassification():

    def __init__(self):
        self.facenet = FaceNet()
        self.pca = PCA(n_components=2)

    def preprocess_image(self ,image_path):
        img = cv2.imread(image_path)
        return img
        #skaliere auf 160x160
        img = cv2.resize(img, (160, 160))
        #normalisierung der pixelintensivität (zwischen 0 und 1)
        img = img.astype("float") / 255.0

        #bild in np-array (shape = (160, 160, 3))
        img = np.array(img, dtype=np.float32)

        #hinzufügen von dimension (shape = (1, 160, 160, 3)) img[0] = Anzahl der Bilder
        img = np.expand_dims(img, axis=0)
        return img

    def get_embeddings(self, image_path_list):
        image_list = [self.preprocess_image(image_path) for image_path in image_path_list]
        return self.facenet.embeddings(image_list)

    def get_all_video_embeddings(self, video_name):
        for face in os.listdir("result"+"/"+video_name):
            print(self.get_embeddings("result"+"/"+video_name+"/"+face).shape)
            print(":::::::::")

    def get_embedded_data(self):
        pass

    def get_classes(self):
        video_faces_path = "result/" + "GX010273/"
        file_list = ["result/" + "GX010273/" + file_name for file_name in os.listdir(video_faces_path)]
        embeddings = self.get_embeddings(file_list)
        all_distances = []
        for face1 in range(len(embeddings)):
            distances = []
            for face2 in range(len(embeddings)):
                distances.append(np.linalg.norm(embeddings[face1]-embeddings[face2]))
            all_distances.append(distances)
        print(tabulate(all_distances,headers = os.listdir(video_faces_path)))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        embeddings = np.array(embeddings[:100])
        x, y, z = embeddings[:, 0], embeddings[:, 1], embeddings[:, 2]
        ax.scatter(x, y, z)

        plt.show()