import cv2
import numpy as np
import json
from deepface import DeepFace
from keras.layers import Dense
from keras_facenet import FaceNet
import os
import sklearn.cluster as cl
from numpy import asarray, expand_dims
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

model = VGGFace(model='senet50', include_top=True, input_shape=(224, 224, 3))
model_list = [model, DeepFace.build_model('DeepID'),DeepFace.build_model('DeepFace'),DeepFace.build_model('OpenFace'),DeepFace.build_model('VGG-Face'),DeepFace.build_model('Facenet')]

class FaceClassification():

    def __init__(self):
        self.facenet = FaceNet()
        self.embedding_data = {}

    def preprocess_img(self,img, shape):
        img = cv2.resize(img,shape)
        img = asarray(img, 'float32')
        img = expand_dims(img, axis=0)
        img = preprocess_input(img, version=2)
        return img
    # extract faces and calculate face embeddings for a list of photo files
    def get_some_embeddings(self, img):
        multiple_way_embed = None
        shape_list = [(224, 224),(47, 55), (152, 152),(96, 96), (224, 224), (160, 160)]
        for model_no in range(len(model_list)):
            embedding = model_list[model_no].predict(self.preprocess_img(img, shape_list[model_no]), verbose=False)
            embedding = embedding / np.linalg.norm(np.array(embedding).flatten(), axis=0,keepdims=True)
            if model_no == 0:
                multiple_way_embed = embedding
            else:
                multiple_way_embed = np.concatenate((multiple_way_embed,embedding), axis=1)

        return multiple_way_embed

    def get_embeddings(self):
        count = 0
        for video_name in os.listdir("result"):
            if os.path.isdir("result/" + video_name):
                for face_img in os.listdir("result/" + video_name):
                    try:
                        img_data = cv2.imread("result/" + video_name + "/" + face_img)
                        img_path = "result/" + video_name + "/" + face_img
                        embedding = self.get_some_embeddings(img_data)
                        embedding = np.array(embedding).flatten() / np.linalg.norm(np.array(embedding).flatten(), axis=0, keepdims=True)
                        self.embedding_data[count] = {"img_name": face_img,
                                                      "video_source": video_name,
                                                      "img_data": cv2.resize(img_data, (160,160)),
                                                      "path": img_path,
                                                      "embedding":embedding}
                        count += 1
                    except cv2.error:
                        print("Invalid frame!")
        img_data_list = [self.embedding_data[img]["img_data"] for img in self.embedding_data]
        embeddings = self.facenet.embeddings(img_data_list)
        for embed in range(len(embeddings)):
            embedding1 = np.array(embeddings[embed])
            embedding1 = embedding1/ np.linalg.norm(embedding1, axis=0, keepdims=True)

            embedding = np.concatenate((embedding1, self.embedding_data[embed]["embedding"]))
            self.embedding_data[embed]["embedding"] = embedding
        return self.embedding_data

    def get_classes(self):
        self.get_embeddings()
        X = np.array([self.embedding_data[face]["embedding"] for face in self.embedding_data])
        X = X.reshape(X.shape[0],-1)
        #X = StandardScaler().fit_transform(X)
        # Erstelle das Dictionary der Hyperparameter-Werte
        labels = self.calculate_labels(X)
        labels_to_img = {}
        for unique_label in np.unique(labels):
            labels_to_img[str(unique_label)] = []
        for data_point in range(len(labels)):
            labels_to_img[str(labels[data_point])].append(self.embedding_data[data_point])

        out_saving_path = "result/labeled/"
        if not os.path.isdir(out_saving_path):
            os.mkdir(out_saving_path)

        for label in labels_to_img:
            different_sources = False
            save_dir = out_saving_path + str(label) +"/"
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            f = open(save_dir + "info.txt", "x")
            f.close()
            f = open(save_dir + "info.txt", "a")
            first_source = labels_to_img[label][0]["video_source"]

            for face in labels_to_img[label]:
                cv2.imwrite(save_dir+face["img_name"],face["img_data"])
                f.write(face["img_name"] + "       video : " + face["video_source"] + "     timestamp : " +
                        face["img_name"].rsplit('.', 1)[0] + " sec\n")
                if face["video_source"] is not first_source:
                    different_sources = True
            f.close()
            if different_sources:
                f = open("result/multiple_sources.txt", "w")
                f.close()
                f = open("result/multiple_sources.txt", "a")
                f.write("face label nr.:" + str(label) + "\n")
                f.close()

    def calculate_labels(self,X):
        eps = 1.1
        dbscan = cl.DBSCAN(eps=eps)
        clusters = dbscan.fit_predict(X)
        return clusters