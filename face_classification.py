import time

import cv2
import numpy as np
import json
from sklearn.cluster import SpectralClustering, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import os
from numpy import asarray, expand_dims
from keras_vggface.utils import preprocess_input
from keras_facenet import FaceNet
import face_recognition
import dlib

class FaceClassification():

    def __init__(self):
        self.facenet = FaceNet()
        self.face_recognition = face_recognition
        self.embedding_data = {}

    def preprocess_img(self,img, shape = None):
        if shape is not None:
            img = cv2.resize(img,shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = asarray(img, 'float32')
        img = expand_dims(img, axis=0)
        img = preprocess_input(img, version=2)
        return img

    def get_embeddings(self):
        if os.path.exists("data.json"):
            print("load data...")
            stored_data = json.load(open("data.json", "r"))
            for face in stored_data:
                embedding = np.load("result/embeddings/"+str(face)+".npy")
                stored_data[face]["embedding"] = embedding
            self.embedding_data = stored_data
            print("data loaded!")

        else:
            if not os.path.isdir("result/embeddings"):
                os.mkdir("result/embeddings")
            print("calculating embedding data...")
            temp_embedding_data = {}
            count = 0
            for folder_name in os.listdir("result"):
                if os.path.isdir("result/" + folder_name) and folder_name != "embeddings":
                    video_name = folder_name
                    for face_img in os.listdir("result/" + video_name):
                        img_path = "result/" + video_name + "/" + face_img
                        img_data = cv2.imread(img_path)
                        if img_data.size == 0:
                            continue
                        height, width, _ = img_data.shape
                        embedding_ = np.array(self.face_recognition.face_encodings(img_data,model="large", num_jitters=1,known_face_locations=[(0,width,height,0)]))
                        if  embedding_.shape[0] != 0:
                            embedding_facenet = np.array(self.facenet.embeddings([img_data]))
                            embedding_ = embedding_.squeeze(axis=0)
                            embedding_facenet = embedding_facenet.squeeze(axis=0)
                            embedding = np.concatenate((embedding_,embedding_facenet),axis=0)
                            np.save("result/embeddings/" + str(count) + ".npy", embedding)
                            img_path = "result/" + video_name + "/" + face_img
                            self.embedding_data[str(count)] = {"img_name": face_img,
                                                               "video_source": video_name,
                                                               "path": img_path,
                                                               "img_data":img_data,
                                                               "embedding":embedding}
                            temp_embedding_data[str(count)] = {"img_name": face_img,
                                                               "video_source": video_name,
                                                               "path": img_path}
                            count += 1
                        else:
                            print("f")
            json.dump(temp_embedding_data, open("data.json", "w"))
            print("embedding data calculated and stored!")

    def get_dlib_embed(self,img_data):
        height, width, _ = img_data.shape
        dlib_embedding = np.array(self.face_recognition.face_encodings(img_data, model="large", num_jitters=1,
                                                                       known_face_locations=[(0, width, height, 0)]))
        dlib_embedding = dlib_embedding.squeeze(axis=0)
        return dlib_embedding
    def get_facenet_embed(self, img_data):
        facenet_embedding = np.array(self.facenet.embeddings([img_data]))
        facenet_embedding = facenet_embedding.squeeze(axis=0)
        return facenet_embedding
    def get_single_embedding(self, img_path):
        img_data = cv2.imread(img_path)
        if not img_data.size == 0:
            dlib_embedding = self.get_dlib_embed(img_data)
            if dlib_embedding.shape[0] != 0:
                facenet_embedding = self.get_facenet_embed(img_data)
                embedding = np.concatenate((dlib_embedding, facenet_embedding), axis=0)
                return embedding
    def get_video_distribution(self,face_dict):


    def all_label(self,X):
        max_k = {}
        gamma = [.6,.65,.675,.75,.77,.8]
        eps_ = [.75,.775,.79,.8,.825,.85,.9]
        n_list = [len(np.unique(np.array(DBSCAN(eps=e).fit_predict(X))))-1 for e in eps_]
        for n in n_list:
            for eps in eps_:
                for g in gamma:
                    clust_spectral = SpectralClustering(n_clusters=n, affinity='rbf', assign_labels='kmeans', eigen_solver="lobpcg", gamma=g, n_init=200).fit_predict(X)
                    clust_dbscan = DBSCAN(eps=eps).fit_predict(X)

                    # Kombiniere die Ergebnisse der drei Clustering-Methoden
                    final_labels = np.zeros(len(X))
                    for i in range(len(X)):
                        if clust_dbscan[i] != -1:
                            final_labels[i] = clust_dbscan[i]
                        else:
                            final_labels[i] = clust_spectral[i] + np.max(clust_dbscan) + 1

                    # Führe abschließend Clustering mit GMM durch
                    gmm = GaussianMixture(n_components=len(np.unique(final_labels)), covariance_type='full')
                    clusters = gmm.fit_predict(X)

                    #clusters = clust_dbscan

                    unique_len = len(np.unique(clusters))

                    if not (unique_len < 2 or unique_len > len(X)-1):
                        #scoring
                        silhouette_avg = silhouette_score(X, clusters)
                        ch_score = calinski_harabasz_score(X, clusters)
                        score = silhouette_avg * ch_score

                        print(eps, silhouette_avg,ch_score,score,unique_len)
                        max_k[str(eps)] = clusters
        return max_k


    def single_video_mass(self):


    def st(self):
        self.get_embeddings()
        X = np.array([self.embedding_data[face]["embedding"] for face in self.embedding_data])
        print(X.shape)
        label_dict = self.all_label(X)
        out_saving_path = "result/labeled/"
        if not os.path.isdir(out_saving_path):
            os.mkdir(out_saving_path)
        for eps in label_dict:
            labels = label_dict[eps]
            labels_to_img = {}
            for unique_label in np.unique(labels):
                labels_to_img[str(unique_label)] = []
            for data_point in range(len(labels)):
                labels_to_img[str(labels[data_point])].append(self.embedding_data[str(data_point)])

            save_dir = out_saving_path + eps
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            for label in labels_to_img:
                save_dir = out_saving_path+ eps +"/" + str(label) + "/"
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                if not os.path.exists(save_dir + "info.txt"):
                    f = open(save_dir + "info.txt", "x")
                    f.close()
                f = open(save_dir + "info.txt", "a")
                first_source = labels_to_img[label][0]["video_source"]

                for face in labels_to_img[label]:
                    image = cv2.imread(face["path"])
                    cv2.imwrite(save_dir + face["img_name"], image)
                    f.write(face["img_name"] + "       video : " + face["video_source"] + "     timestamp : " + face["img_name"].rsplit('.', 1)[0] + " sec\n")
                f.close()


    def get_classes(self):
        self.get_embeddings()
        X = np.array([self.embedding_data[face]["embedding"] for face in self.embedding_data])
        X = X.reshape(X.shape[0],-1)
        #X = StandardScaler().fit_transform(X)
        # Erstelle das Dictionary der Hyperparameter-Werte
        labels = self.all_label(X)
        labels_to_img = {}
        for unique_label in np.unique(labels):
            labels_to_img[str(unique_label)] = []
        for data_point in range(len(labels)):
            labels_to_img[str(labels[data_point])].append(self.embedding_data[str(data_point)])

        out_saving_path = "result/labeled/"
        if not os.path.isdir(out_saving_path):
            os.mkdir(out_saving_path)

        for label in labels_to_img:
            different_sources = False
            save_dir = out_saving_path + str(label) +"/"
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            if not os.path.exists(save_dir + "info.txt"):
                f = open(save_dir + "info.txt", "x")
                f.close()
            f = open(save_dir + "info.txt", "a")
            first_source = labels_to_img[label][0]["video_source"]

            for face in labels_to_img[label]:
                cv2.imwrite(save_dir+face["img_name"],cv2.imread(face["path"]))
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
