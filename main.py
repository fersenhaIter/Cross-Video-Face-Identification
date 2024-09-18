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
        if not os.path.isdir("result"):
            os.mkdir("result")

        recent_videos = os.listdir("result")
        if len(recent_videos) != 0:
            last_video = max(recent_videos, key = lambda video: os.path.getmtime("result/"+video), default="")
            recent_videos.remove(last_video)

        for file in os.listdir(directory):
            dir = directory + "/" + file
            if os.path.isdir(dir):
                self.run_data_preparation(dir)
            else:
                filename = os.fsdecode(file)
                name = filename.rsplit('.', 1)[0]
                file_format = filename.rsplit('.', 1)[1]

                if name in recent_videos:
                    print(f"skip: {name}")
                    continue

                if file_format in ["LRV","MP4","mp4"]:
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

cam_analysis = CamAnalysis()
#cam_analysis.run_data_preparation("C:/Users/jakob/Downloads/gkd_4jakob_2023-03-30_1342")
cam_analysis.face_classification.st()
