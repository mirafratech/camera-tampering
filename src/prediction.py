# Written : Suraj Goswami
# EmpID : 4093
# Mirafra Technologies
# Version : 1.0

import numpy as np
import pandas as pd
import cv2
from src.cnn_prediction import CnnPredict
from src.image_processing import ImageProcessing


class Predict:

    def __init__(self, model_path,video_file):
        self.cnn = CnnPredict(model_path)
        self.video = video_file
        self.img_proc = ImageProcessing()

    def csv_file(self,frame_num, pred_lst):
        # Saving a csv file for batch of predicted images

        dict1 = {"Frame_num": frame_num, "Pred_class": pred_lst}

        df = pd.DataFrame(dict1)  # Creating a dataframe
        df.to_csv("files/camera_tampering.csv", index=False)  # Saving the dataframe in csv format

        print("Csv file created as 'files/camera_tempering.csv'")

    def predict(self):
        # Video Frame
        cap = cv2.VideoCapture(self.video)
        # Background Substraction method
        fgbg = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=16,
                                                  detectShadows=False)  # generating a foreground mask
        kernel = np.ones((5, 5), np.uint8)

        pred_class = []
        pred_lst = []
        i = 0
        frame_num = []

        while True:
            ret, frame = cap.read()  # reading all the frames
            if ret:
                pred = self.img_proc.image_processing(frame, kernel, fgbg)  # Here 3 = Moved , 0= NotMoved

                if pred == 0:
                    pred = self.cnn.cnn_arc(frame)

                pred_class.append(pred)

                if len(pred_class) > 70:  #Skip top 70 frames
                    pred_class.pop(0)

                if all(i > 0 for i in pred_class):
                    # cv2.putText(frame, "Tamper: ", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print("Tampering Detected")
                    if all(i == 3 for i in pred_class):
                        pred = 3
                        # cv2.putText(frame, "Moved", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    elif all(i == 2 for i in pred_class):
                        pred = 2
                        # cv2.putText(frame, "Defocussed", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    else:
                        pred = 1
                        # cv2.putText(frame, "Covered", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    pred = 0
                    # cv2.putText(frame, "Normal", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                pred_lst.append(pred)
                frame_num.append(i)
                i+=1
                if cv2.waitKey(30) & 0xff == ord('q'):  # To close the camera press q
                    break
            else:
                break
        self.csv_file(frame_num, pred_lst)







