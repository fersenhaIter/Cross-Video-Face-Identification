import shutil

import cv2
from keras_facenet import FaceNet
import numpy as np
import os
import sklearn.cluster as cl
from sklearn.metrics import silhouette_score

class FaceClassification():

    def __init__(self):
        self.facenet = FaceNet()
        self.embedding_data = {}

    import numpy as np
    import cv2

    def align_face(image):
        # Wähle das Gesichtserkennungsmodell deiner Wahl (z.B. OpenCV Haar Cascade)
        face_cascade = cv2.CascadeClassifier('path/to/haarcascade_frontalface_default.xml')

        # Wandle das Bild in Graustufen um, um die Gesichtserkennung zu erleichtern
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Wende die Gesichtserkennung auf das Bild an
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Wenn kein Gesicht gefunden wurde, gib None zurück
        if len(faces) == 0:
            return None

        # Wenn mehr als ein Gesicht gefunden wurde, wähle das größte
        biggest_face = max(faces, key=lambda face: face[2] * face[3])

        # Extrahiere die Koordinaten des größten Gesichts
        x, y, w, h = biggest_face

        # Schneide das größte Gesicht aus dem Bild aus
        face = image[y:y + h, x:x + w]

        # Wandle das ausgeschnittene Gesicht in Graustufen um
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Verwende einen Gesichtslandmark-Erkennungsalgorithmus, um die Gesichtslandmarks im Gesicht zu finden (z.B. dlib)
        # Extrahiere die Koordinaten der Augen aus den Landmarks
        left_eye = (0, 0)
        right_eye = (0, 0)

        # Berechne die Rotation, die erforderlich ist, um die Augen horizontal auszurichten
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Berechne das Zentrum des Gesichts
        center = (x + w // 2, y + h // 2)

        # Erstelle die Transformationsmatrix, um das Gesicht um das Zentrum und den Winkel zu drehen
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Wende die Transformationsmatrix auf das Gesicht an
        aligned_face = cv2.warpAffine(face, M, (w, h), flags=cv2.INTER_CUBIC)

        return aligned_face

    def get_embeddings(self):
        count = 0
        for video_name in os.listdir("result"):
            if os.path.isdir("result/"+video_name):
                for face_img in os.listdir("result/"+video_name):
                    self.embedding_data[count] = {"img_name":face_img,
                                                  "video_source":video_name,
                                                  "img_data":cv2.imread("result/"+video_name+"/"+face_img),
                                                  "path":"result/"+video_name+"/"+face_img}
                    count+=1
        img_list = [self.align_face(self.embedding_data[face_img]["img_data"]) for face_img in self.embedding_data]
        embedding = self.facenet.embeddings(img_list)
        for face_img_nr in range(len(self.embedding_data)):
            self.embedding_data[list(self.embedding_data.keys())[face_img_nr]]["embedding"] = embedding[face_img_nr]

        return self.embedding_data

    def get_classes(self):
        self.get_embeddings()
        X = np.vstack([self.embedding_data[embed]["embedding"] for embed in self.embedding_data])
        # Erstelle das Dictionary der Hyperparameter-Werte
        eps = [0.3,0.4,0.5, 0.6, 0.7,0.75, 0.79,0.8,0.85,0.9]
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



    def calculate_k(self, X, eps):
        max_k = {"score": -1}
        # Berechne den Silhouetten-Score für k-Werte zwischen 2 und 10
        for k in range(2, len(self.embedding_data)):

            cluster = cl.DBSCAN(eps=eps)
            cluster.fit(X)

            labels = cluster.fit_predict(X)

            score = silhouette_score(X, cluster.labels_)
            if score >= max_k["score"]:
                max_k = {"k": k, "score": score, "labels": labels}
        return max_k