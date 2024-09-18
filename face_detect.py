import os.path
from datetime import datetime

import cv2
import time
import numpy as np
from mtcnn import MTCNN
from Retinaface.Retinaface import FaceDetector

class FaceDetection():

    def __init__(self,min_face_size = 40, scan_frame_rate = 3, video_name = "video.mp4", downgraded_fps = 9.6):
        #variables
        self.count = 0
        self.skipped_counter = 0
        self.skipped_counter1 = 0
        self.current_frame = None
        self.min_face_size = min_face_size
        self.scan_frame_rate = scan_frame_rate
        self.downgraded_fps = downgraded_fps
        self.video_name = video_name
        self.starting_point = 0
        self.video_faces = {}

        #models
        self.mtcnn_face_detector = MTCNN(scale_factor = .4, min_face_size = self.min_face_size, steps_threshold = [0.4, 0.5, 0.6])
        self.retina_face_detector = FaceDetector(confidence_threshold=.94, name="mobilenet", top_k = 3000, keep_top_k = 500)

    def get_faces(self):
        face_name = str(round(self.count / self.downgraded_fps, 2))
        faces = self.retina_face_detector.detect_align(self.current_frame,threshold=.99)
        for i,face in enumerate(faces):
            self.video_faces[face_name+str(i)] = np.array(face)

    def store_image(self, img_data, face_name):
        cv2.imwrite("result/"+ self.video_name + "/" + face_name + ".jpg", img_data)

    def check_frame_similarity(self,next_frame):

        if (self.current_frame is None):
            return True
        #start = time.perf_counter()
        # brightness
        next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        next_mean_brightness = np.mean(next_frame_gray) / 255
        current_frame_gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)

        # https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html --> param
        # flow for (x,y) stored in third dimension
        flow = cv2.calcOpticalFlowFarneback(cv2.resize(next_frame_gray, (900, 900)),
                                            cv2.resize(current_frame_gray, (900, 900)), None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # get x-, y-components
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        magnitude, angle = cv2.cartToPolar(flow_x, flow_y, angleInDegrees=True)
        flow_mean = np.mean(magnitude) / 10
        #end = time.perf_counter()
        #print(f"similarity dpf: {end - start}")
        cv2.putText(next_frame, f'brightness:{next_mean_brightness:.2f}', (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        cv2.putText(next_frame, f'flow      :{flow_mean:.2f}', (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(next_frame, f'skipped   :{self.skipped_counter}', (50, 155), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2)
        if ((flow_mean < .65 and next_mean_brightness > 0.25 and self.skipped_counter != 3) or (
                flow_mean < 0.35 and next_mean_brightness < 0.1 and self.skipped_counter != 7)):
            self.skipped_counter += 1
            self.skipped_counter1 += 1
            return False
        self.skipped_counter = 0
        return True

    def get_video_frame_faces(self,video_path, starting_point = 0):

        vid = cv2.VideoCapture(video_path)
        mean_frame_duration = []
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vid.get(cv2.CAP_PROP_FPS)
        frames = range(int(starting_point*fps), total_frames, int(fps/9.6))

        self.starting_point = starting_point
        self.count = int(starting_point * fps)
        self.video_name = os.path.basename(video_path).rsplit('.', 1)[0]
        self.skipped_count = 0

        print("\n")
        print(datetime.now().strftime("%H:%M:%S"))
        print(f"total: {len(frames)}")
        for frame_number in frames:
            start = time.perf_counter()

            self.count += 1
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = vid.read()
            if ret and self.check_frame_similarity(frame):
                self.current_frame = frame
                self.get_faces()
            else:
                self.current_frame = frame
            end = time.perf_counter()
            mean_frame_duration.append(end-start)

        vid.release()
        print(f"skipped : {self.skipped_counter1}")
        print(f"mean dpf: {sum(mean_frame_duration)/len(mean_frame_duration)}")
        print(datetime.now().strftime("%H:%M:%S"))
        print("\n")
