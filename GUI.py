import json

import pandas as pd
from joblib import load
from kivy.uix.boxlayout import BoxLayout
from kivymd.app import MDApp
from sklearn.preprocessing import LabelEncoder, RobustScaler


class LoadModel:
    def __init__():
        model = load('D:\OneDrive\Academia\MSc Machine Learning in Science\Modules\PHYS4038 Scientific Programming in Python\Submissions\model.joblib')

class GUI(MDApp):
    def __init__(self, **kwargs):
        super(GUI, self).__init__(**kwargs)  #not working - creating two instances of GUI() on launch in the same windows??

    def build(self):
        #with open('D:\OneDrive\Academia\MSc Machine Learning in Science\Modules\PHYS4038 Scientific Programming in Python\Submissions\col_names.json', 'rb') as fp:
            #data_cats = json.load(fp)
            #gui_data = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', '']

        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Orange"

        #not working - doesn't give selectable categtories on GUI
        self._running_app.root.ids.dashboard.ids.cat_gender.items = self.gui_cats.gender.unique()
        self._running_app.root.ids.dashboard.ids.cat_hypertension.items = self.gui_cats.hypertension.unique()
        self._running_app.root.ids.dashboard.ids.cat_heart_disease.items = self.gui_cats.heart_disease.unique()
        self._running_app.root.ids.dashboard.ids.cat_ever_married.items = self.gui_cats.ever_married.unique()
        self._running_app.root.ids.dashboard.ids.cat_employment.items = self.gui_cats.work_type.unique()
        self._running_app.root.ids.dashboard.ids.cat_residence.items = self.gui_cats.Residence_type.unique()
        self._running_app.root.ids.dashboard.ids.cat_smoking_status.items = self.gui_cats.smoking_status.unique()

        self.x = 0

    def preprocess(self, data):
        data = data.dropna(axis = 0)
        data = data.reset_index()
        data['age'] = data['age'].astype(int)

        #future work - these should be saved and imported from training
        encoder = LabelEncoder()
        scaler = RobustScaler()

        for i in data.columns:
            if data[i].dtype == 'object':
                data[i] = encoder.fit_transform(data[i])
            elif data[i].dtype == 'float64':
                data[i] = scaler.fit_transform(data[[i]])
            else:
                pass

                return data

    def predict(self):
        model = load('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/PHYS4038 Scientific Programming in Python/Submissions/model.joblib')
        features = json.load(open('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/PHYS4038 Scientific Programming in Python/Submissions/features.json'))
        data_types = json.load(open('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/PHYS4038 Scientific Programming in Python/Submissions/data_types.json'))
        df = pd.DataFrame(columns = features)

        #for column in df:
        #    for dtype in data_types:
        #        df.astype({df.iloc[column]:data_types[dtype]})

        #future work - refactor attempting to use the above nested loop
        df.astype({'gender':'object'})
        df.astype({'age':'int32'})
        df.astype({'hypertension':'int64'})
        df.astype({'heart_disease':'int64'})
        df.astype({'ever_married':'object'})
        df.astype({'work_type':'object'})
        df.astype({'Residence_type':'object'})
        df.astype({'avg_glucose_level':'float64'})
        df.astype({'bmi':'float64'})
        df.astype({'smoking_status':'object'})

        #future work - eliminate further inefficiencies below
        user_data = ['Male', 67, 0, 1, 'Yes', 'Private', 'Urban', 228.69, 36.6, 'formerly smoked'] #take from GUI user input, currently placeholder info
        df_data0 = {'gender':user_data[0], 'age':user_data[1], 'hypertension':user_data[2], 'heart_disease':user_data[3], 'ever_married':user_data[4], 'work_type':user_data[5], 'Residence_type':user_data[6], 'avg_glucose_level':user_data[7], 'bmi':user_data[8], 'smoking_status':user_data[9]}
        df_data1 = {'gender':user_data[0], 'age':user_data[1] + 5, 'hypertension':user_data[2], 'heart_disease':user_data[3], 'ever_married':user_data[4], 'work_type':user_data[5], 'Residence_type':user_data[6], 'avg_glucose_level':user_data[7], 'bmi':user_data[8], 'smoking_status':user_data[9]}
        df_data2 = {'gender':user_data[0], 'age':user_data[1] + 10, 'hypertension':user_data[2], 'heart_disease':user_data[3], 'ever_married':user_data[4], 'work_type':user_data[5], 'Residence_type':user_data[6], 'avg_glucose_level':user_data[7], 'bmi':user_data[8], 'smoking_status':user_data[9]}
        df = df.append(df_data0, ignore_index = True)
        df = df.append(df_data1, ignore_index = True)
        df = df.append(df_data2, ignore_index = True)

        df = self.preprocess(df.iloc[:, 1:11])
        output = pd.DataFrame(model.predict(df), columns = ['probability']) #dtype error produced
        print(output.head())

    def restart(): #future work
        pass

    def exit(): #future work
        pass

class Dashboard(BoxLayout):
    pass

def main():
    GUI().run()

if __name__ == '__main__':
    main()