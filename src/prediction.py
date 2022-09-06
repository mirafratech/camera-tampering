# Written : Suraj Goswami
# EmpID : 4093
# Mirafra Technologies
# Version : 1.0

import tensorflow as tf
import numpy as np
from keras.models import load_model
from src.training import Training
import os
import pandas as pd


class Predict:
    def __init__(self,  model_path, test_path, image_size):

        self.predict_path = test_path  # Path of the batch of images to predict
        self.model_path = model_path  # Model path
        self.model = load_model(self.model_path)  # Loading Model
        self.image_size = image_size   # Image size

    def predict(self, image):

        # Prediction for single image

        image = tf.keras.preprocessing.image.load_img(image, target_size=(self.image_size,self.image_size))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch
        input_arr = input_arr.astype('float32') / 255.
        predictions = self.model.predict(input_arr)
        predicted_class = np.argmax(predictions, axis=-1)
        return predicted_class[0]  # Predicted class would be in 2D

    def csv_file(self):
        # Saving a csv file for batch of predicted images

        image_path = []
        actual_class = []
        pred_class = []

        for main_fol in os.listdir(self.predict_path):  # In test path there will be 4 folders 0,1,2,3
            for img_file in os.listdir(os.path.join(self.predict_path, main_fol)):  # Path for images

                path = os.path.join(self.predict_path, main_fol, img_file)  # Creating image path to save in csv

                image_path.append(path)  # Appending image path in list
                actual_class.append(int(main_fol))  # Append True class 0,1,2,3 as they are in folder format

                predicted_class = self.predict(path)  # Predicting the image

                pred_class.append(int(predicted_class))  # Appending the predict result for that image

        dict1 = {"Image_path": image_path, "Class": actual_class, "Pred_class": pred_class}

        df = pd.DataFrame(dict1)  # Creating a dataframe
        df.to_csv("files/camera_tampering.csv", index=False)  # Saving the dataframe in csv format

        print("Csv file created as 'files/camera_tempering.csv'")





