import face_detect
import face_classification
import os


class CamAnalysis:
    def __init__(self, save_dir):
        self.face_detection = face_detect.FaceDetection()
        self.face_classification = face_classification.FaceClassification()
        self.save_dir = save_dir
        self.data = {}
        self.video_names = ["20220428_131159"]

    def run_data_preparation(self, directory):
        print(directory)
        for file in os.listdir(directory):
            dir = directory + "/" + file
            if os.path.isdir(dir):
                self.run_data_preparation(dir)
                break
            filename = os.fsdecode(file)
            name = filename.rsplit('.', 1)[0]
            save_dir = self.save_dir + name
            self.video_names.append(name)
            if(not os.path.isdir(save_dir)):
                os.mkdir(save_dir)
            print(filename)
            self.face_detection.get_video_frame_faces(dir, save_dir)

    def run_face_classification(self):
        for video in self.video_names:
            video_faces_path = self.save_dir + video
            for face in os.listdir(video_faces_path):
                timestamp = face.rsplit('.', 1)[0]
                self.data[len(self.data)] = {"timestamp":timestamp, "file":video, "embeddings":self.face_classification.get_embeddings(video_faces_path + "/" + face)}

cam_analysis = CamAnalysis("C:/Users/jakob/Downloads/gkd_4jakob_2023-03-30_1342/faces/")
cam_analysis.run_data_preparation("C:/Users/jakob/Downloads/gkd_4jakob_2023-03-30_1342/4jakob")
cam_analysis.run_face_classification()
cam_analysis.run_data_preparation("C:/Users/jakob/Downloads/gkd_4jakob_2023-03-30_1342/4jakob")
