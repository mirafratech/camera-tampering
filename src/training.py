
# Written : Suraj Goswami
# EmpID : 4093
# Mirafra Technologies
# Version : 1.0

from keras.models import Model
from keras.layers import Input,Dropout,Flatten,Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam,SGD,Adadelta
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.callbacks import ModelCheckpoint
import scipy
import glob
import os


class Training:
    def __init__(self, train_path, valid_path,  image_size, batch_train, batch_valid, output_path, model_name):
        self.train_path = train_path    # Image path for training, should be in directory format
        self.valid_path = valid_path   # Image path for validation
        self.output_path = output_path
        self.img_size = image_size
        self.train_len = len(glob.glob(self.train_path + "/**/*.jpg", recursive=True))  # Total no. of train images
        self.valid_len = len(glob.glob(self.valid_path + "/**/*.jpg", recursive=True))  # Total no. of test images
        self.batch_size_train = batch_train  # Multiple of no. of images
        self.batch_size_valid = batch_valid
        self.model_path = output_path
        self.model_name = model_name

    def image_generator(self):

        train_datagen = ImageDataGenerator(rescale=1/255.)
        valid_datagen = ImageDataGenerator(rescale=1/255.)

        train_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size_train,
            class_mode='categorical'
        )

        valid_generator = valid_datagen.flow_from_directory(
            self.valid_path,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size_valid,
            class_mode='categorical'
        )

        return train_generator, valid_generator

    def pretrianed_model(self, model_name):
        # Transfer Learning using VGG-16 model
        if model_name == "VGG16":
            model = VGG16(include_top=False, weights='imagenet', input_shape=(self.img_size, self.img_size, 3))
        else:
            model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(self.img_size, self.img_size, 3))

        return model

    def add_layers(self):
        # Adding 1 each of flatten,dense, dropout layers
        # Number of classes 4: Normal,Covered,Defocussed,Moved

        nb_classes = 4
        model = self.pretrianed_model(self.model_name)
        input1 = Input(shape=(self.img_size, self.img_size, 3), name='image_input')
        output_model = model(input1)
        x = Flatten(name='flatten')(output_model)
        x = Dense(64, activation='relu', name='fc1')(x)
        x = Dropout(0.25, name="dropout1")(x)
        output = Dense(nb_classes, activation="softmax", name="output")(x)

        model = Model(inputs=input1, outputs=output)

        return model

    def training(self):
        # Model training will start

        model = self.add_layers()
        train_generator, valid_generator = self.image_generator()
        epochs = 50                        # Epoch is less as it was taking huge time
        lr = 0.005
        decay_rate = lr / epochs
        optimizer = Adadelta(lr=lr, rho=0.95, epsilon=1e-08, decay=decay_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # filepath = Path of the model to be saved
        checkpointer = ModelCheckpoint(filepath=self.model_path,
                                       verbose=1, save_best_only=True)
        callbacks_list = [checkpointer]

        steps_epoch = self.train_len / self.batch_size_train
        valid_steps = self.valid_len / self.batch_size_valid

        model.fit_generator(
            train_generator,
            steps_per_epoch=steps_epoch,
            epochs=epochs,
            callbacks=callbacks_list,
            validation_data=valid_generator,
            validation_steps=valid_steps
        )

        print(f"Model is Trained as {self.output_path+'weights.best.trained.hdf5'} ")

