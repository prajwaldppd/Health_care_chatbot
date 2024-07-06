import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import csv
import warnings
import tkinter as tk
from tkinter import messagebox, simpledialog

warnings.filterwarnings("ignore", category=DeprecationWarning)


training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

reduced_data = training.groupby(training['prognosis']).max()

# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)

scores = cross_val_score(clf, x_test, y_test, cv=3)
print(scores.mean())

model = SVC()
model.fit(x_train, y_train)
print("for svm: ")
print(model.score(x_test, y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

severityDictionary = {}

with open('MasterData/symptom_severity.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) == 2:
            symptom = row[0]
            severity = int(row[1])
            severityDictionary[symptom] = severity

description_list = {}

with open('MasterData/symptom_Description.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        description_list[row[0]] = row[1]

precautionDictionary = {}

with open('MasterData/symptom_precaution.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]


def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []


def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))


def tree_to_code(tree, feature_names, disease_input, num_days):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])

            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]

            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = tk.messagebox.askquestion("Symptom Check", f"Are you experiencing {syms}?")
                if inp == "yes":
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days)

            if present_disease[0] == second_prediction[0]:
                result_text.set(f"You may have {present_disease[0]}\n\nDescription:\n{description_list[present_disease[0]]}")
            else:
                result_text.set(f"You may have {present_disease[0]} or {second_prediction[0]}\n\nDescription:\n{description_list[present_disease[0]]}\n{description_list[second_prediction[0]]}")

            precution_list = precautionDictionary[present_disease[0]]
            precautions.set("\n".join([f"{i + 1}) {j}" for i, j in enumerate(precution_list)]))

    recurse(0, 1)


def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum = sum + severityDictionary[item]
    if (sum * days) / (len(exp) + 1) > 13:
        messagebox.showinfo("Consultation", "You should take the consultation from doctor.")
    else:
        messagebox.showinfo("Precaution", "It might not be that bad but you should take precautions.")


def get_symptoms():
    name = name_var.get()
    disease_input = symptom_var.get()
    try:
        num_days = int(days_var.get())
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid number of days.")
        return
    if not name or not disease_input:
        messagebox.showerror("Missing Information", "Please enter all the details.")
        return

    conf, cnf_dis = check_pattern(cols, disease_input)
    if conf == 1:
        if len(cnf_dis) == 1:
            disease_input = cnf_dis[0]
        else:
            selected_symptom = simpledialog.askinteger("Select Symptom", f"Multiple matches found. Select one (0 to {len(cnf_dis)-1}):",
                                                       initialvalue=0, minvalue=0, maxvalue=len(cnf_dis) - 1)
            disease_input = cnf_dis[selected_symptom]
    else:
        messagebox.showerror("Invalid Symptom", "Please enter a valid symptom.")
        return

    tree_to_code(clf, cols, disease_input, num_days)


app = tk.Tk()
app.title("HealthCare ChatBot")

name_var = tk.StringVar()
symptom_var = tk.StringVar()
days_var = tk.StringVar()
result_text = tk.StringVar()
precautions = tk.StringVar()

tk.Label(app, text="-----------------------------------HealthCare ChatBot-----------------------------------", font=("Arial", 16)).pack()

tk.Label(app, text="Your Name:").pack()
tk.Entry(app, textvariable=name_var).pack()

tk.Label(app, text="Enter the symptom you are experiencing:").pack()
tk.Entry(app, textvariable=symptom_var).pack()

tk.Label(app, text="From how many days?").pack()
tk.Entry(app, textvariable=days_var).pack()

tk.Button(app, text="Submit", command=get_symptoms).pack()

tk.Label(app, textvariable=result_text, wraplength=500, justify="left").pack()
tk.Label(app, text="Precautions:", font=("Arial", 12, "bold")).pack()
tk.Label(app, textvariable=precautions, wraplength=500, justify="left").pack()

app.mainloop()
