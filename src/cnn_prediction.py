
import cv2
import tensorflow as tf
import numpy as np


class CnnPredict:
    def __init__(self,  model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def cnn_arc(self, frame):

        img_shape = (self.input_details[0]['shape'][1], self.input_details[0]['shape'][2])  # (224,224)

        img_size = cv2.resize(frame, img_shape, interpolation=cv2.INTER_NEAREST)
        img_size = np.reshape(img_size, tuple(self.input_details[0]['shape']))

        img_size = img_size / 255.

        img_size = np.float32(img_size)

        self.interpreter.set_tensor(self.input_details[0]['index'], img_size)

        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        classes = np.argmax(output_data, axis=-1)

        if classes[0] == 1:  # Covered
            pred = 1
        elif classes[0] == 2:  # Defocussed
            pred = 2
        else:
            pred = 0  # Normal

        return pred