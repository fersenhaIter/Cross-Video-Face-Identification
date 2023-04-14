import os.path

import cv2
import face_detection
from datetime import datetime
import numpy as np
from mtcnn import MTCNN

class FaceDetection():

    def __init__(self,min_face_size = 40, scan_frame_rate = 3, video_name = "video.mp4"):
        #variables
        self.count = 0
        self.skipped_count = 0
        self.current_frame = None
        self.min_face_size = min_face_size
        self.scan_frame_rate = scan_frame_rate
        self.fps = 29/self.scan_frame_rate
        self.video_name = video_name
        self.recent_frame_faces = []

        #models
        self.mtcnn_face_detector = MTCNN(min_face_size = self.min_face_size)
        self.retina_net_res_net = face_detection.build_detector("RetinaNetResNet50", confidence_threshold=.5, nms_iou_threshold=.3)

    def overlapping_area(self, p0 ,p1 ,p2 ,p3, get_IoU = False):
        x, y = 0,1
        x_left = max(p0[x], p2[x])
        y_top = max(p0[y], p2[y])
        x_right = min(p1[x], p3[x])
        y_bottom = min(p1[y], p3[y])

        area = 0
        if x_left >= x_right and y_top >= y_bottom:
            return None

        overlap_area = (x_right-x_left)*(y_bottom-y_top)

        if(get_IoU):
            area_1 = (p1[x]-p0[x])*(p1[y]-p0[y])
            area_2 = (p3[x]-p2[x])*(p3[y]-p2[y])
            union = area_1 + area_2 - overlap_area
            return overlap_area/union

        x_left = min(p0[x], p2[x])
        y_top = min(p0[y], p2[y])
        x_right = max(p1[x], p3[x])
        y_bottom = max(p1[y], p3[y])

        max_rect_area = (x_right-x_left)*(y_bottom-y_top)
        relative_overlapping = overlap_area/max_rect_area

        if(relative_overlapping>0.5): return (x_left,y_top),(x_right,y_bottom)

        return None

    def draw_facial_landmarks(self, keypoints):
        cv2.circle(self.current_frame, keypoints["left_eye"], radius=1, color=(0, 0, 255), thickness=1)
        cv2.circle(self.current_frame, keypoints["right_eye"], radius=1, color=(0, 0, 255), thickness=1)
        cv2.circle(self.current_frame, keypoints["nose"], radius=1, color=(0, 0, 255), thickness=-1)
        cv2.line(self.current_frame, keypoints["mouth_left"], keypoints["mouth_right"], color=(0, 0, 255), thickness=1)

    def check_same_face(self,current_mtcnn_faces):
        '''IoUs = []
        print(boxes_current_frame)
        print(self.recent_frame_faces)'''
        for recent_face in self.recent_frame_faces:
            removable = True
            p2, p3 = recent_face
            for current_face in current_mtcnn_faces:
                (p0,p1) = current_mtcnn_faces[current_face]["box"]
                IoU = self.overlapping_area(p0,p1, p2, p3, True)
                if IoU is None:
                    break
                #IoUs.append(IoU)
                if IoU >= 0.6:
                    print("duplicate!")
                    current_mtcnn_faces.pop(current_face)
                    removable = False
            if removable:
                self.recent_frame_faces.remove(recent_face)
        '''print(boxes_current_frame)
        print(IoUs)'''
        return current_mtcnn_faces

    def get_faces(self):
        frame_faces = {}
        self.recent_frame_faces = []

        mtcnn_faces =  [detected_face for detected_face in self.mtcnn_face_detector.detect_faces(self.current_frame) if detected_face["confidence"] >= 0.95]
        mtcnn_faces =  [detected_face for detected_face in mtcnn_faces if detected_face["box"][2]>= self.min_face_size and detected_face["box"][3]>= self.min_face_size]
        for face in mtcnn_faces:
            x, y, w, h = face["box"]
            face["box"] = ((x, y), (x + w, y + h))
        reduced_mtcnn_faces = self.check_same_face(mtcnn_faces)
        if (len(reduced_mtcnn_faces) == 0 ):
            return frame_faces
        retina_faces = self.retina_net_res_net.detect(self.current_frame).astype(int)
        for face in reduced_mtcnn_faces:
            (p0,p1) = face["box"]
            for (x1, y1, x2, y2, score) in retina_faces:
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = max(x2, 0)
                y2 = max(y2, 0)
                #cv2.rectangle(self.current_frame,(x1,y1),(x2,y2), (0, 255, 0), 1)
                if x2 - x1 > self.min_face_size and y2 - y1 > self.min_face_size:
                    rect = self.overlapping_area(p0, p1, (x1, y1), (x2, y2))
                    if (rect is None):
                        break
                    (x1, y1), (x2, y2) = rect
                    self.recent_frame_faces.append(((x1, y1), (x2, y2)))
                    face_name = str(round(self.count / self.fps, 2))

                    while (frame_faces.keys().__contains__(face_name)):
                        face_name += "0"
                    print("add face")
                    frame_faces[face_name]["rect"] = ((int(x1), int(y1)), (int(x2), int(y2)))
                    frame_faces[face_name]["keypoints"] = face["keypoints"]
        return frame_faces

    def align_face(self, face):
        (x1, y1), (x2, y2) = face["rect"]
        left_eye = face["keypoints"]["left_eye"]
        right_eye = face["keypoints"]["right_eye"]
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        #rotationmatrix
        (h, w) = self.current_frame.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        #new rect-points
        rect_points = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], dtype=np.float32)
        rect_points = np.expand_dims(rect_points, axis=1)
        rect_points = cv2.transform(rect_points, M)
        rect_points = np.squeeze(rect_points, axis=1)

        # check new bounding points
        if np.any(rect_points < 0) or np.any(rect_points[:, 0] > self.current_frame.shape[1]) or np.any(
                rect_points[:, 1] > self.current_frame.shape[0]):
            # adjust bounding points
            rect_points[:, 0] = np.clip(rect_points[:, 0], 0, self.current_frame.shape[1])
            rect_points[:, 1] = np.clip(rect_points[:, 1], 0, self.current_frame.shape[0])

        # transform image through rotationmatrix
        aligned_face = cv2.warpAffine(self.current_frame, M, (self.current_frame.shape[1], self.current_frame.shape[0]))

        #new outcut
        roi = aligned_face[rect_points[0][1]:rect_points[2][1],rect_points[0][0]:rect_points[1][0]]

        return roi

    def store_image(self, frame_faces):
        for face in frame_faces:
            # Region of Interest
            roi = self.align_face(face)
            cv2.imwrite("result/"+ self.video_name + "/" + face + ".jpg", roi)

    def check_frame_similarity(self,next_frame):
        if(self.current_frame is None):
            return True

        #brightness
        next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        next_mean_brightness = np.mean(next_frame_gray)/255
        current_frame_gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)

        #https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html --> param
        #flow for (x,y) stored in third dimension
        flow = cv2.calcOpticalFlowFarneback(next_frame_gray, current_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #get x-, y-components
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        magnitude, angle = cv2.cartToPolar(flow_x, flow_y, angleInDegrees=True)
        flow_mean = np.mean(magnitude)/10

        if ((flow_mean < 1.4 and next_mean_brightness > 0.3 and self.skipped_count < 3) or (flow_mean < 0.2 and next_mean_brightness < 0.1 and self.skipped_count < 6)):
            self.skipped_count += 1
            return False
        self.skipped_count = 0
        return True

    def get_video_frame_faces(self,video_path, starting_point = 0):
        print("start:")
        print(datetime.now().strftime("%H:%M:%S"))
        print(f"starting at:{str(starting_point)}")

        self.video_name = os.path.basename(video_path).rsplit('.', 1)[0]
        self.count = int(starting_point * self.fps)
        vid = cv2.VideoCapture(video_path)
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_number in range(starting_point*29, total_frames, self.scan_frame_rate):
            self.count += 1
            print(self.count)
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = vid.read()
            if ret and self.check_frame_similarity(frame):
                self.current_frame = frame
                frame_faces = self.get_faces()
                #cv2.imshow("",self.current_frame)
                if len(frame_faces) != 0:
                    self.store_image(frame_faces)
                '''if cv2.waitKey(1) == ord('q'):
                    break'''
            else:
                print("to similar")
                self.current_frame = frame

        # After the loop release the cap object
        vid.release()
        print("end:")
        print(datetime.now().strftime("%H:%M:%S"))