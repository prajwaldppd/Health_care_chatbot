import re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import csv
import warnings
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load and preprocess data
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']
reduced_data = training.groupby(training['prognosis']).max()

# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier().fit(x_train, y_train)
svm_clf = SVC().fit(x_train, y_train)

# Loading symptom severity, descriptions, and precautions
severityDictionary = {}
with open('MasterData/symptom_severity.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) == 2:
            symptom, severity = row[0], int(row[1])
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
        precautionDictionary[row[0]] = row[1:5]

# Symptom checker
def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    return (1, pred_list) if len(pred_list) > 0 else (0, [])

def sec_predict(symptoms_exp):
    input_vector = np.zeros(len(cols))
    for item in symptoms_exp:
        if item in cols:
            input_vector[cols.get_loc(item)] = 1
    return clf.predict([input_vector])

def calc_condition(exp, days):
    severity_sum = sum(severityDictionary[item] for item in exp)
    if (severity_sum * days) / (len(exp) + 1) > 13:
        messagebox.showinfo("Consultation", "You should take the consultation from a doctor.")
    else:
        messagebox.showinfo("Precaution", "It might not be that bad but you should take precautions.")
        
        
# here we are using decision tree 
# yaha input of 5 symptoms le rahe hai 


def DecisionTree():
    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    l2 = [0] * len(cols)
    for symptom in psymptoms:
        if symptom in cols:
            l2[cols.get_loc(symptom)] = 1

    inputtest = [l2]
    predict = clf.predict(inputtest)
    predicted = predict[0]
    disease_name = le.inverse_transform([predicted])[0]

    t1.delete("1.0", tk.END)
    t1.insert(tk.END, disease_name)

def SVM():
    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    l2 = [0] * len(cols)
    for symptom in psymptoms:
        if symptom in cols:
            l2[cols.get_loc(symptom)] = 1

    inputtest = [l2]
    predict = svm_clf.predict(inputtest)
    predicted = predict[0]
    disease_name = le.inverse_transform([predicted])[0]

    t2.delete("1.0", tk.END)
    t2.insert(tk.END, disease_name)

# GUI setup
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Disease Predictor")
    root.geometry("900x600")

    # Load and set background image
    bg_image = Image.open("bg3.jpg")
    bg_image = bg_image.resize((900, 600), Image.Resampling.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Heading
    w2 = tk.Label(root, justify=tk.CENTER, text="Disease Predictor using Machine Learning", fg="black", bg='SystemButtonFace')
    w2.config(font=("Helvetica", 24, "bold"))
    w2.pack(pady=20)

    # Input frame
    input_frame = tk.Frame(root, bg='SystemButtonFace')
    input_frame.pack(pady=20)

    # Labels and entries
    entries = [
        ("Symptom 1", Symptom1 := tk.StringVar()),
        ("Symptom 2", Symptom2 := tk.StringVar()),
        ("Symptom 3", Symptom3 := tk.StringVar()),
        ("Symptom 4", Symptom4 := tk.StringVar()),
        ("Symptom 5", Symptom5 := tk.StringVar()),
    ]

    for i, (text, var) in enumerate(entries):
        label = tk.Label(input_frame, text=text, fg="black", bg='SystemButtonFace', font=("Helvetica", 12))
        label.grid(row=i, column=0, pady=5, sticky=tk.W)
        entry = ttk.Combobox(input_frame, textvariable=var, values=sorted(cols), font=("Helvetica", 12))
        entry.grid(row=i, column=1, pady=5, padx=10)

    # Button frame
    button_frame = tk.Frame(root, bg='SystemButtonFace')
    button_frame.pack(pady=20)

    # Buttons
    buttons = [
        ("DecisionTree", DecisionTree),
        ("SVM", SVM)
    ]

    for text, command in buttons:
        btn = tk.Button(button_frame, text=text, command=command, bg="#4CAF50", fg="white", font=("Helvetica", 12), padx=10, pady=5)
        btn.pack(side=tk.LEFT, padx=10)

    # Result frame
    result_frame = tk.Frame(root, bg='SystemButtonFace')
    result_frame.pack(pady=20)

    # Results
    results = [
        ("DecisionTree Result:", t1 := tk.Text(result_frame, width=40, height=1, font=("Helvetica", 12), state='normal')),
        ("SVM Result:", t2 := tk.Text(result_frame, width=40, height=1, font=("Helvetica", 12), state='normal'))
    ]

    for i, (text, t) in enumerate(results):
        tk.Label(result_frame, text=text, fg="black", bg='SystemButtonFace', font=("Helvetica", 12)).grid(row=i, column=0, pady=5, sticky=tk.W)
        t.grid(row=i, column=1, pady=5, padx=10)

    root.mainloop()
 