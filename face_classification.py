import shutil

import cv2
import face_recognition
from facenet.src.facenet import load_model
from keras_facenet import FaceNet
import numpy as np
import os
from sklearn.model_selection import GridSearchCV
import sklearn.cluster as cl
from sklearn.metrics import silhouette_score

class FaceClassification():

    def __init__(self):
        self.facenet = FaceNet()
        self.embedding_data = {}

    def preprocess_image(self ,image_path):
        img = cv2.imread(image_path)
        #skaliere auf 160x160
        img = cv2.resize(img, (160, 160))
        #normalisierung der pixelintensivität (zwischen 0 und 1)
        img = img.astype("float") / 255.0

        #bild in np-array (shape = (160, 160, 3))
        img = np.array(img, dtype=np.float32)

        #hinzufügen von dimension (shape = (1, 160, 160, 3)) img[0] = Anzahl der Bilder
        img = np.expand_dims(img, axis=0)
        return img

    def get_embedding(self, img_path):

        image = cv2.imread(img_path)

        # Variante 1: Gesamtbild-Embedding
        #full_face_embedding = face_recognition.face_encodings(image)

        # Variante 2: Multi-Task Cascaded Convolutional Neural Network
        mtcnn_face_embedding = face_recognition.face_encodings(image, model='small')

        # Variante 3: Deep Residual Network
        #dlib_face_embedding = face_recognition.face_encodings(image, model='large')

        #combined_embedding = np.hstack((full_face_embedding, mtcnn_face_embedding, dlib_face_embedding))
        return mtcnn_face_embedding

    def get_embeddings(self):
        count = 0
        for video_name in os.listdir("result"):
            if os.path.isdir("result/"+video_name):
                for face_img in os.listdir("result/"+video_name):
                    self.embedding_data[count] = {"img_name":face_img, "video_source":video_name, "img_data":cv2.imread("result/"+video_name+"/"+face_img), "path":"result/"+video_name+"/"+face_img}
                    count+=1
        deleting_keys = []
        for face_nr in self.embedding_data:
            self.embedding_data[face_nr]["embedding"] = self.get_embedding(self.embedding_data[face_nr]["path"])
            if np.array(self.embedding_data[face_nr]["embedding"]).shape[0] == 0:
                deleting_keys.append(face_nr)
        for key in deleting_keys:
            self.embedding_data.pop(key)
        return self.embedding_data

    def get_classes(self):
        self.get_embeddings()
        X = np.vstack([self.embedding_data[embed]["embedding"] for embed in self.embedding_data])
        # Erstelle das Dictionary der Hyperparameter-Werte
        eps = [0.3,0.4,0.5, 0.6, 0.7,0.75, 0.79,0.8,0.9]
        for eps_ in eps:
            k = self.calculate_k(X, eps_)
            if not k.keys().__contains__("labels"):
                break
            labels = k["labels"]
            for data_point in range(len(self.embedding_data.keys())):
                if not os.path.isdir("result/clustered_"+"EPS"+str(eps_)):
                    os.mkdir("result/clustered_"+"EPS"+str(eps_))
                save_dir = "result/clustered_"+"EPS"+str(eps_)+"/"+str(labels[data_point])
                self.embedding_data[list(self.embedding_data.keys())[data_point]]["label"] = labels[data_point]
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                cv2.imwrite(save_dir+"/"+self.embedding_data[list(self.embedding_data.keys())[data_point]]["img_name"], self.embedding_data[list(self.embedding_data.keys())[data_point]]["img_data"])


    # function returns WSS score for k values from 1 to kmax
    def calculate_k(self, X, eps):
        max_k = {"score": -1}
        # Berechne den Silhouetten-Score für k-Werte zwischen 2 und 10
        for k in range(2, len(self.embedding_data)):

            cluster = cl.DBSCAN(eps=eps)
            cluster.fit(X)

            labels = cluster.fit_predict(X)

            # Überprüfen, ob mindestens zwei Cluster gefunden wurden
            n_clusters = len(set(labels))
            if n_clusters < 2:
                continue

            score = silhouette_score(X, cluster.labels_)
            if score >= max_k["score"]:
                max_k = {"k": k, "score": score, "labels": labels}
        return max_k