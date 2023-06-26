import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix
from yellowbrick.model_selection import FeatureImportances
import graphviz


data = pd.read_csv('diabetes-dataset.csv')
print(data)


class_names = ["non-diabetes", "diabetes"]


X = data.drop(columns = 'Outcome')
Y = data['Outcome'].astype(str)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


clf = DecisionTreeClassifier(max_depth=20, random_state=42)
clf = clf.fit(X_train,Y_train)


clf.get_params()


predictions = clf.predict(X_test)


viz = FeatureImportances(clf)
viz.fit(X_test,Y_test)
viz.show()


y_pred = clf.predict(X_test)
visualizer = ClassificationReport(clf, support=True)
visualizer.fit(X_train, Y_train)
visualizer.score(X_test, Y_test)
visualizer.show()


cm = ConfusionMatrix(clf)
cm.score(X_test, Y_test)
cm.show()


dot_data = export_graphviz(clf, out_file=None, filled=True, rounded=True, special_characters=True, class_names=class_names)
graph = graphviz.Source(dot_data)
graph.render("diabetes_decision_tree", format='png')
graph.view()


accuracy = accuracy_score(Y_test, predictions)
print("Analisys completed with success!")
print("________________________________")
print(f"Accuracy: {accuracy*100:.3f}%")
print("________________________________")
print("Now you will be able to predict if you have diabetes or not!")
print("________________________________")
print("Please, enter your information below:")
print("________________________________")
pregnancies = int(input("Pregnancies: "))
glucose = int(input("Glucose: "))
blood_pressure = int(input("Blood Pressure: "))
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
    print("You don't have diabetes.")
else:
    print("You have diabetes.")
