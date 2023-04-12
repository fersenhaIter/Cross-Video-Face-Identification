import cv2
from keras_facenet import FaceNet
import numpy as np
import os

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



class FaceClassification():

    def __init__(self):
        self.facenet = FaceNet()
        self.embedding_data = {}

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

    def get_embeddings(self):
        count = 0
        for video_name in os.listdir("result"):
            if os.path.isdir("result/"+video_name):
                for face_img in os.listdir("result/"+video_name):
                    self.embedding_data[count] = {"img_name":face_img, "video_source":video_name}
                    count+=1
        image_list = [cv2.imread("result/"+self.embedding_data[image]["video_source"]+"/"+self.embedding_data[image]["img_name"]) for image in self.embedding_data]
        all_embeddings = self.facenet.embeddings(image_list)
        for face_nr in self.embedding_data:
            self.embedding_data[face_nr]["embedding"] = all_embeddings[face_nr]
        return self.embedding_data

    def get_classes(self):
        self.get_embeddings()
        print(self.calculate_WSS())

    # function returns WSS score for k values from 1 to kmax
    def calculate_WSS(self):
        X = [self.embedding_data[embed]["embedding"] for embed in self.embedding_data]
        scores = {}
        # Berechne den Silhouetten-Score für k-Werte zwischen 2 und 10
        for k in range(2, len(self.embedding_data)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(X)
            score = silhouette_score(X, kmeans.labels_)
            scores[k]=score
            print("Silhouette score for k=%d: %0.4f" % (k, score))
        return list(scores.keys())[list(scores.values()).index(max(scores.values()))]