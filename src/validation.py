# Written : Suraj Goswami
# EmpID : 4093
# Mirafra Technologies
# Version : 1.0

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score


class Valid:
    def __init__(self, csv_path):
        self.csv_path = csv_path  # Path to csv file

    def valid_matrix(self):
        df = pd.read_csv(self.csv_path)
        y_valid = np.array(df['Class'])  # Features present in csv file
        y_pred = np.array(df['Pred_class'])

        print(metrics.classification_report(y_valid, y_pred))

        # Average weight is taken as the images in each classes may not be balanced
        print("\nf1_score :", f1_score(y_valid, y_pred, average='weighted'))
        print("\nprecision : ", precision_score(y_valid, y_pred, average='weighted'))
        print("\nrecall :", recall_score(y_valid, y_pred, average='weighted'))
