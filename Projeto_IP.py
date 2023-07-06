import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix
from yellowbrick.model_selection import FeatureImportances
import graphviz

data = pd.read_csv(r'/home/mateus/Desktop/Estudo/VSCode Projects/DiabetesDetection/diabetes-detection/diabetes-dataset.csv')

## Tratamento dos dados, com base em valores que não podem ser 0
treated_data = data.loc[(data['Glucose'] != 0) & (data['BloodPressure'] != 0) &	(data['SkinThickness'] != 0) & 
                        (data['Insulin'] != 0) & (data['BMI'] != 0.0) & (data['DiabetesPedigreeFunction'] != 0.0) &	(data['Age'] != 0)]

simplified_data = data.drop(columns = ['Insulin', 'DiabetesPedigreeFunction'])

class_names = ["non-diabetes", "diabetes"]

## Criação da árvore de decisão
X = treated_data.drop(columns = 'Outcome')
Y = treated_data['Outcome'].astype(str)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

clf = DecisionTreeClassifier(max_depth=20, random_state=42)
clf = clf.fit(X_train,Y_train)

## Árvore de decisão com informações reduzidas
X_user = simplified_data.drop(columns = 'Outcome')
Y_user = simplified_data['Outcome'].astype(str)
X_user_train, X_user_test, Y_user_train, Y_user_test = train_test_split(X_user, Y_user, test_size=0.3)

user_clf = DecisionTreeClassifier(max_depth=20, random_state=42)
user_clf = user_clf.fit(X_user_train,Y_user_train)

predictions = clf.predict(X_test)
user_predictions = user_clf.predict(X_user_test)


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
    print("________________________________")


def test_pacient():
    print("Now you will be able to predict the probabilty of you having diabetes, based on this model.")
    print("________________________________")
    print("Please, enter your information below:")
    print("________________________________")
    pregnancies = int(input("Pregnancies: "))
    glucose = int(input("Glucose: "))
    blood_pressure = int(input("Diastolic Blood Pressure: "))
    skin_thickness = int(input("Skin Thickness: "))
    height = float(input("Height(cm): "))/100
    weight = float(input("Weight(Kg): "))
    bmi = weight/(height**2)
    age = int(input("Age: "))
    print("________________________________")

    individual_info = [pregnancies, glucose, blood_pressure, skin_thickness, bmi, age]
    example_data = pd.DataFrame([individual_info], columns = ['Pregnancies', 'Glucose',	'BloodPressure',	'SkinThickness',	'BMI',	'Age'])
      
    prediction = user_clf.predict(example_data)
    print(prediction)
    
    if prediction == 1:
        print("According to our classification, you possibly have diabetes, consult a doctor. ")
    else: 
        print("According to out classification, you possibly don't have diabetes.")
        

def main():
    print("Chose one of the options below, by typing the correspondent number: \n1: See Feature Importance Graph\n"
          "2: See Classification Report\n3: See Connfusion Matrix\n4: See Decision Tree\n5: Check the classification's accuracy\n"
          "6: Check classification for an individual's data\n\"E\": Exit program")
    option = input().strip().upper()

    match option:
        case '1': 
            feature_importance()
            main()
        case '2':
            classification_report()
            main()
        case '3':
            confusion_matrix()
            main()
        case '4':
            decision_tree()
            main()
        case '5':
            get_accuracy()
            main()
        case '6':
            test_pacient()
            main()
        case 'E':
            print("\033[34mThanks for using this program!\033[m")
        case _:
            print("\033[31mInvalid Entrance!\033[m")
            main()


if __name__ == '__main__':
    main()
