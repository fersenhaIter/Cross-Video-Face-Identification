import face_detect
import face_classification
import os


class CamAnalysis:
    def __init__(self):
        self.face_detection = face_detect.FaceDetection()
        self.face_classification = face_classification.FaceClassification()
        self.data = {}
        self.video_names = []

    def most_recent_frame(self, dir):
        last_sec = max([float(face.rsplit('.', 1)[0]) for face in os.listdir(dir)])
        return int(last_sec)


    def run_data_preparation(self, directory):
        print(directory)
        for file in os.listdir(directory):
            dir = directory + "/" + file
            if os.path.isdir(dir):
                self.run_data_preparation(dir)
            else:
                filename = os.fsdecode(file)
                name = filename.rsplit('.', 1)[0]
                save_dir = "result/" + name
                self.video_names.append(name)
                print(filename)
                if(not os.path.isdir(save_dir)):
                    os.mkdir(save_dir)
                    self.face_detection.get_video_frame_faces(dir)
                elif(os.listdir(save_dir).__len__() != 0):
                    most_recent_sec = self.most_recent_frame(save_dir)
                    self.face_detection.get_video_frame_faces(dir, starting_point=most_recent_sec)
                else:
                    self.face_detection.get_video_frame_faces(dir, starting_point=0)


    def run_face_classification(self):
        for video in self.video_names:
            video_faces_path = "result/" + video
            for face in os.listdir(video_faces_path):
                timestamp = face.rsplit('.', 1)[0]
                self.data[len(self.data)] = {"timestamp":timestamp, "file":video, "embeddings":self.face_classification.get_embeddings(video_faces_path + "/" + face)}

cam_analysis = CamAnalysis()
cam_analysis.run_data_preparation("C:/Users/jakob/Downloads/gkd_4jakob_2023-03-30_1342/4jakob")
