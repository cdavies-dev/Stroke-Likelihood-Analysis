# Scientific Programming in Python – Submission 1

## Project: Predicting the Likelihood of a Stroke Using an Artificial Neural Network

**Name:** Charlie Davies
**Course:** MSc Machine Learning in Science
**University:** University of Nottingham
**Student number:** 20462957

## Background

The idea for the project is to use an artificial neural network, like a Multilayer Perceptron, to classify whether a person is likely to have a stroke based on the following:

1. Gender
2. Age
3. Hypertension (high blood pressure)
4. Existing Heart Disease
5. Marital Status
6. Employment Status
7. Residential Status
8. Average Glucose
9. Body Mass Index (BMI)
10. Smoking Status

The result of a combination of the above features will either be a 1 (stroke likely) or a 0 (stroke not likely). This will be displayed via an interactive GUI built in Python. Hopefully, with enough time, an attempt to predict at what age the individual is likely to have a stroke based on a 1 result can be implemented.

## Modules and Other Sources

With some preparation and insight into the requirements for the project, it is likely that the following libraries will be used:

- PyQt5: https://pypi.org/project/PyQt6/
- Keras: https://pypi.org/project/keras/
- Matplotlib: https://pypi.org/project/matplotlib/
- Pandas: https://pypi.org/project/pandas2/
- Numpy: https://pypi.org/project/numpy/
- Seaborn: https://pypi.org/project/seaborn/

Dataset source: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

## Code Structure

The project aims to adhere roughly to PEP 8 (https://peps.python.org/pep-0008/) guidelines. The main structure will centre around Object Orientation and the use of multiple files for concision, readability and reusability.

Currently, three classes exist with some basic code. These are:

```
Preprocess:
  __init__()
  data_manager()
  normalise()
NeuralNetwork:
  __init__()
GUI:
  __init__()
main()
```

Once a relatively sensible-looking output is achieved, these classes will be split into three separate ```.py``` files and GUI development will begin.

## Current Implementation

The following code has been implemented and is working to a certain degree.

```
class Preprocess:
    def __init__(self, csv):
        self.train_X, self.train_y, self.test_X, self.test_y = self.data_manager(csv)

    def data_manager(self, csv):
        #import file, drop rows containing N/A
        input = pd.read_csv(csv, header = 0).dropna(axis = 0)
        input['gender'].replace(['Male', 'Female', 'Other'], [0, 1, 2], inplace = True) #remove only one example of 2 for simplicity?
        input['ever_married'].replace(['Yes', 'No'], [0, 1], inplace = True)
        input['work_type'].replace(['Private', 'Self-employed', 'Govt_job', 'Never_worked', 'children'], [0, 1, 2, 3, 4], inplace = True)
        input['Residence_type'].replace(['Urban', 'Rural'], [0, 1], inplace = True)
        input['smoking_status'].replace(['formerly smoked', 'never smoked', 'smokes', 'Unknown'], [0, 1, 2, 3], inplace = True)
        input.reset_index()
        
        #70% training data, normalise features, split class
        training_data = input.sample(frac = 0.7, random_state = 200)
        train_X = self.normalise(training_data.iloc[:, 1:11])
        train_y = training_data.iloc[:, 11:12]
        
        #30% testing data, normalise features, split class
        testing_data = input.drop(training_data.index)
        test_X = self.normalise(testing_data.iloc[:, 1:11])
        test_y = testing_data.iloc[:, 11:12]

        return train_X, train_y, test_X, test_y

    def normalise(self, input_features):
        #
        # this needs only to take into account continuous values that aren't age
        #
        
        x = input_features

        for i in x.columns:
            x[i] = (x[i] - x[i].min()) / (x[i].max() - x[i].min())

        return x
```

## Next Steps

The following pseudocode includes the next steps for the project and an insight into the class/method structure.

```
class NeuralNetwork: #mlp?
    #__init__()
        #take training and testing sets
        #training()
        #testing()

    #training()
        #model = Sequential()...
            #rest of architecture definition
        #complete model   

    #testing(model)
        #store results

    #plots()
        #
        # investigate if and which plots would be useful here
        #

    #
    # investigate how to deploy this for use in a live app with the GUI
    #

    pass

class GUI: #PyQt6? Kivy?
    #__init__()
        #take the deployed model
        #run the GUI
        
    #run(model)
        #provide assumption on new input data

    #restart()

    #exit()
    
    #
    # investigate putting this class in another file for organisation
    #

    pass
```

## Scope and Uncertainties

Currently, the module used to build the GUI is not confirmed. Depending on the complexity involved with implementing the module, this may need to change when approaching the deadline as it isn't the main focus of the project. Further to this, calculating the age at which an individual is likely to have a stroke may stretch beyond the timescale of the project. The primary concern is building a classifier which makes a prediction given unseen user inputs.

## Files

Please find the code file pre-commit and relevant dataset attached.
