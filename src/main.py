# Written : Suraj Goswami
# EmpID : 4093
# Mirafra Technologies
# Version : 1.0

from src.training import Training
from src.prediction import Predict
# from src.validation import Valid
import argparse
import pathlib

if __name__ == "__main__":

    def training(train_path, valid_path, image_size, batch_train,batch_valid, output_path, model_train):

        pathlib.Path(output_path).parents[0].mkdir(parents=True, exist_ok=True)
        obj_train = Training(train_path, valid_path, image_size, batch_train, batch_valid, output_path, model_train)
        obj_train.training()

    def predict(model_path, video_file):
        obj_pred = Predict(model_path, video_file)
        obj_pred.predict()

    # def validation(csv_path):
    #     obj_valid = Valid(csv_path)
    #     obj_valid.valid_matrix()

    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("Model",
                        nargs="?",
                        choices=['training', 'predict'],
                        default='training',
                        )
    args, sub_args = parser.parse_known_args()

    if args.Model == "training":
        parser = argparse.ArgumentParser()
        parser.add_argument('-train_path', type=str, required=True)
        parser.add_argument('-valid_path', type=str, required=True)
        parser.add_argument('-image_size',  type=int, default=256)
        parser.add_argument('-batch_train',  type=int, default=50)
        parser.add_argument('-batch_valid',  type=int, default=50)
        parser.add_argument('-output_path', type=str, default="/model/")
        parser.add_argument('-model_train', type=str, default="MobileNetV2")
        args = parser.parse_args(sub_args)
        training(args.train_path, args.valid_path,args.image_size, args.batch_train, args.batch_valid, args.output_path,
                 args.model_train)
    elif args.Model == "predict":
        parser = argparse.ArgumentParser()
        parser.add_argument('-model_path', type=str, default="model/converted_model15.tflite")
        parser.add_argument('-video_path', type=str)
        args = parser.parse_args(sub_args)
        predict(args.model_path, args.video_path)
    # elif args.Model == "validation":
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('-csv_path', type=str, default="files/camera_tampering.csv")
    #     args = parser.parse_args(sub_args)
    #     validation(args.csv_path)

