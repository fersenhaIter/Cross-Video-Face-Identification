import face_detect
import face_classification
import os


class CamAnalysis:
    def __init__(self, save_dir):
        self.face_detection = face_detect.FaceDetection()
        self.face_classification = face_classification.FaceClassification()
        self.save_dir = save_dir

    def run_data_preparation(self, directory):
        print(directory)
        for file in os.listdir(directory):
            potential_new_dir = os.path.join(directory, file)
            if os.path.isdir(potential_new_dir):
                self.run_data_preparation(potential_new_dir)
                break
            filename = os.fsdecode(file)
            name = filename.rsplit('.', 1)[0]
            save_dir = self.save_dir + name
            os.mkdir(save_dir)
            self.face_detection.get_video_frame_faces(directory + "/" + filename, save_dir)

    def run_face_classification(self):
        pass

cam_analysis = CamAnalysis("C:/Users/jakob/Downloads/gkd_4jakob_2023-03-30_1342/faces/")
cam_analysis.run_data_preparation("C:/Users/jakob/Downloads/gkd_4jakob_2023-03-30_1342/4jakob")
