import cv2
import numpy as np
import time

class DistrModule():

    def __init__(self) -> None:
        self.time_offs = 5
        self.start_time = -1
        self.blink_time = 0.2
        self.last_sight = time.time()
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.filterByArea = True
        detector_params.maxArea = 1500
        self.detector = cv2.SimpleBlobDetector_create(detector_params)


    def detect_faces(self, img, cascade):
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
        if len(coords) > 1:
            biggest = (0, 0, 0, 0)
            for i in coords:
                if i[3] > biggest[3]:
                    biggest = i
            biggest = np.array([i], np.int32)
        elif len(coords) == 1:
            biggest = coords
        else:
            return None
        for (x, y, w, h) in biggest:
            frame = img[y:y + h, x:x + w]
        return frame


    def detect_eyes(self, img, cascade):
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
        width = np.size(img, 1)  # get face frame width
        height = np.size(img, 0)  # get face frame height
        left_eye = None
        right_eye = None
        for (x, y, w, h) in eyes:
            if y > height / 2:
                pass
            eyecenter = x + w / 2  # get the eye center
            if eyecenter < width * 0.5:
                # print(f"LEFT EYE: {(x, y, w, h)}")
                left_eye = img[y:y + h, x:x + w]
            else:
                # print(f"RIGHT EYE: {(x, y, w, h)}")
                right_eye = img[y:y + h, x:x + w]
        # print(left_eye)
        # print(right_eye)

        if left_eye is not None and right_eye is not None:  # both eyes are open
            self.last_sight = time.time()   # initializes last time that both eyes are open.
        elif time.time() - self.last_sight >= self.blink_time:
            print(f"Time passed since last sight: {time.time() - self.last_sight}")
            return None, None

        # returns None, None if both eyes are closed.
        return left_eye, right_eye


    def cut_eyebrows(self, img):
        height, width = img.shape[:2]
        eyebrow_h = int(height / 4)
        img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)

        return img


    def blob_process(self, img, threshold, detector):
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
        img = cv2.erode(img, None, iterations=2)
        img = cv2.dilate(img, None, iterations=4)
        img = cv2.medianBlur(img, 5)
        keypoints = detector.detect(img)
        # print(keypoints)
        return keypoints


    @staticmethod
    def nothing(x):
        pass


    def main(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('image')
        cv2.createTrackbar('threshold', 'image', 0, 255, DistrModule.nothing)
        while True:
            eye_detected = False
            _, frame = cap.read()
            face_frame = self.detect_faces(frame, self.face_cascade)
            if face_frame is not None:
                eyes = self.detect_eyes(face_frame, self.eye_cascade)
                for eye in eyes:
                    if eye is not None:
                        eye_detected = True
                        self.start_time = -1
                        threshold = r = cv2.getTrackbarPos('threshold', 'image')
                        eye = self.cut_eyebrows(eye)
                        keypoints = self.blob_process(eye, threshold, self.detector)
                        eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # print(eye_detected)
            if not eye_detected:
                if self.start_time <= 0:
                    self.start_time = time.time()
                elif time.time() - self.start_time >= self.time_offs:
                    print("ALERT: DRIVER ASLEEP")
            cv2.imshow('image', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    m = DistrModule()
    m.main()
