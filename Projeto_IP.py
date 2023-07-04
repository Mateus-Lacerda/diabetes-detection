import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix
from yellowbrick.model_selection import FeatureImportances
from tkinter import *
from PIL import Image, ImageDraw
from tkinter import messagebox
import graphviz

data = pd.read_csv('diabetes-dataset.csv')

## Tratamento dos dados, com base em valores que não podem ser 0
treated_data = data.loc[(data['Glucose'] != 0) & (data['BloodPressure'] != 0) &	(data['SkinThickness'] != 0) & 
                        (data['Insulin'] != 0) & (data['BMI'] != 0.0) & (data['DiabetesPedigreeFunction'] != 0.0) &	(data['Age'] != 0)]

class_names = ["non-diabetes", "diabetes"]

X = treated_data.drop(columns = 'Outcome')
Y = treated_data['Outcome'].astype(str)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

clf = DecisionTreeClassifier(max_depth=20, random_state=42)
clf = clf.fit(X_train,Y_train)

predictions = clf.predict(X_test)


def feature_importance():
    viz = FeatureImportances(clf)
    viz.fit(X_test,Y_test)
    viz.show()


def classification_report():
    visualizer = ClassificationReport(clf, support=True)
    visualizer.fit(X_train, Y_train)
    visualizer.score(X_test, Y_test)
    visualizer.show()


def confusion_matrix():
    cm = ConfusionMatrix(clf)
    cm.score(X_test, Y_test)
    cm.show()


def decision_tree():
    dot_data = export_graphviz(clf, out_file=None, filled=True, rounded=True, special_characters=True, class_names=class_names)
    graph = graphviz.Source(dot_data)
    graph.render("diabetes_decision_tree", format='png')
    graph.view()


def get_accuracy():
    accuracy = accuracy_score(Y_test, predictions)
    print("Analisys completed with success!")
    print("________________________________")
    print(f"Accuracy: {accuracy*100:.3f}%")


def test_pacient():
    print("Now you will be able to predict if you have diabetes or not!")
    print("________________________________")
    print("Please, enter your information below:")
    print("________________________________")
    pregnancies = int(input("Pregnancies: "))
    glucose = int(input("Glucose: "))
    blood_pressure = int(input("Diastolic Blood Pressure: "))
    skin_thickness = int(input("Skin Thickness: "))
    insulin = int(input("Insulin: "))
    bmi = float(input("BMI: "))
    diabetes_pedigree_function = float(input("Diabetes Pedigree Function: "))
    age = int(input("Age: "))
    print("________________________________")

    individual_info = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age])
    example_data = pd.DataFrame([individual_info], columns = ['Pregnancies', 'Glucose',	'BloodPressure',	'SkinThickness',
                                                            'Insulin',	'BMI',	'DiabetesPedigreeFunction',	'Age'])
    predictions_2 = clf.predict(example_data)

    if predictions_2 == '0':
        print("Your info indicate that you probably don't have diabetes. ")
    else:
        print("Your sympthons show signs of diabetes, consult a doctor. ")
