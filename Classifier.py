import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler


class Preprocessing:
    def __init__(self, dataset):
        clean_data = self.data_cleaning(dataset)
        X = clean_data.iloc[:, 1:11] #to be overwritten, only for dumping dtypes below
        json.dump(list(X.dtypes.astype(str)), open('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/PHYS4038 Scientific Programming in Python/Submissions/data_types.json', 'w'))
        
        norm_data = self.data_normalisation(clean_data)
    
        X = norm_data.iloc[:, 1:11]
        y = norm_data.iloc[:, 11:12]

        json.dump([column for column in X], open('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/PHYS4038 Scientific Programming in Python/Submissions/features.json', 'w'))

        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, test_size = 0.3, random_state = 200)

    def data_cleaning(self, data):
            data = data.dropna(axis = 0)
            data = data.drop('id', axis = 1)
            data = data.reset_index()
            data['age'] = data['age'].astype(int)
            
            return data

    def data_normalisation(self, data):
        encoder = LabelEncoder()
        scaler = RobustScaler()

        #future work - update to save and output encoder and scaler for testing data
        for i in data.columns:
            if data[i].dtype == 'object':
                data[i] = encoder.fit_transform(data[i])
            elif data[i].dtype == 'float64':
                data[i] = scaler.fit_transform(data[[i]])
            else:
                pass

        return data
     
    def visualisations(self, data): #not called as not necessary
        data = data.drop('index', axis = 1) 

        #correlation matrix
        sb.heatmap(data.corr(), annot = True)
        plt.show()

        #feature, output histograms
        for col in data:
            sb.histplot(data[col])
            plt.show()

class BuildModel:
    def __init__(self):
        data = Preprocessing(pd.read_csv('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/PHYS4038 Scientific Programming in Python/Submissions/stroke.csv', header = 0))
    
        model = LogisticRegression(
        penalty = 'l2',
        random_state = 42,
        solver = 'saga',
        max_iter = 7000)

        scores = cross_validate(model, data.train_X, np.ravel(data.train_y), cv = 5, scoring = 'accuracy', return_train_score = True)
        model.fit(data.train_X, np.ravel(data.train_y))
        
        print('\nTraining Accuracy: {}%, 5-Fold Cross-Validation Accuracy: {}%'.format(round(np.mean(scores['train_score']) * 100, 2), round(np.mean(scores['test_score']) * 100, 2)))
        print('Testing Accuracy: {}%\n'.format(round(model.score(data.test_X, np.ravel(data.test_y)) * 100, 2)))

        dump(model, 'D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/PHYS4038 Scientific Programming in Python/Submissions/model.joblib')

        self.confusion_matrix(model, data.test_X, data.test_y)
    
    def confusion_matrix(self, model, test_X, test_y):
        pred_y = model.predict(test_X)
        cf_matrix = confusion_matrix(test_y, pred_y, labels = model.classes_)
        display = ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = model.classes_)
        display.plot()
        plt.show()

def main():
    BuildModel()

if __name__ == '__main__':
    main()