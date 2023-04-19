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
model1 = DeepFace.build_model('DeepID')
model2 = DeepFace.build_model('DeepFace')
model3 = DeepFace.build_model('OpenFace')
model4 = DeepFace.build_model('VGG-Face')
model5 = DeepFace.build_model('Facenet')

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
        model_list = [model, model1, model2, model3, model4, model5]
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
        label_list = self.calculate_labels(X)
        for eps_ in label_list:
            labels = label_list[eps_]
            labels_to_img = {}
            for unique_label in np.unique(labels):
                labels_to_img[str(unique_label)] = []
            for data_point in range(len(labels)):
                labels_to_img[str(labels[data_point])].append(self.embedding_data[data_point])

            out_saving_path = "result/clustered_" + "EPS_" + str(round(eps_,3))
            if not os.path.isdir(out_saving_path):
                os.mkdir(out_saving_path)

            for label in labels_to_img:
                different_sources = False
                save_dir = out_saving_path + "/" + str(label) +"/"
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                for face in labels_to_img[label]:
                    cv2.imwrite(save_dir+"/"+face["img_name"],face["img_data"])
                f = open(save_dir + "info.txt", "x")
                f.close()
                f = open(save_dir + "info.txt", "a")
                first_source = labels_to_img[label][0]["video_source"]
                for img in labels_to_img[label]:
                    cv2.imwrite(save_dir + "/" + img["img_name"], np.array(img["img_data"]))
                    f.write(img["img_name"] + "       video : " + img["video_source"] + "     timestamp : " +
                            img["img_name"].rsplit('.', 1)[0] + " sec\n")
                    if img["video_source"] is not first_source:
                        different_sources = True
                f.close()
    def calculate_k(self, X):
        max_k = {}
        for k in range(2, len(X)-1):
            kmeans = cl.KMeans(n_clusters=k)
            clusters = kmeans.fit_predict(X)
            '''silhouette_avg = silhouette_score(X, clusters)

            # Calinski-Harabasz Score berechnen
            ch_score = calinski_harabasz_score(X, clusters)
            score = silhouette_avg*ch_score'''

            max_k[k] = clusters
        return max_k

    def calculate_labels(self,X):
        max_k = {}
        # Berechne den Silhouetten-Score für k-Werte zwischen 2 und 10
        eps = 0.1
        while len(np.unique(cl.DBSCAN(eps=eps).fit_predict(X))) == 1:
            eps+=0.1
            print(eps, len(np.unique(cl.DBSCAN(eps=eps).fit_predict(X))))
        unique = len(np.unique(cl.DBSCAN(eps=eps).fit_predict(X)))
        while 1 < unique < len(X):
            dbscan = cl.DBSCAN(eps=eps)
            clusters = dbscan.fit_predict(X)

            silhouette_avg = silhouette_score(X, clusters)

            # Calinski-Harabasz Score berechnen
            ch_score = calinski_harabasz_score(X, clusters)

            score = silhouette_avg * ch_score

            print(eps, silhouette_avg,ch_score,score, unique)

            max_k[eps] = clusters
            eps += 0.1
            unique = len(np.unique(cl.DBSCAN(eps=eps).fit_predict(X)))
        return max_k