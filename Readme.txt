
#Training using VGG16 model
python main.py training -train_path="output10/train" -valid_path="output10/valid"

#Predicting model on batch of images
python main.py predict -test_path="output10/test"

#Evaluating model- Getting F1_score, precision,recall
python main.py validation

#Check argument parser main.py files for more variables


