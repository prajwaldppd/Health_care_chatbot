### Disease Predictor using Machine Learning
Welcome to the Disease Predictor project! This application utilizes machine learning algorithms to predict potential diseases based on the symptoms provided by the user. The graphical user interface (GUI) is built using Tkinter and allows users to enter symptoms and get predictions from both Decision Tree and Support Vector Machine (SVM) models.

## 📝 Description
The Disease Predictor application leverages machine learning to analyze symptoms and predict possible diseases. The app uses two models: Decision Tree and SVM. Based on the input symptoms, it lists all potential diseases along with their probabilities and provides precautionary measures.

## ⚙️ Installation
To run this application, you need to have Python installed on your machine. Follow the steps below to set up the environment and install the necessary dependencies.

## Prerequisites
1. Python 3.x
2. pip (Python package installer)
    Steps
3. **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/disease-predictor.git
    cd disease-predictor

4. Create a virtual environment (optional but recommended):
    ```bash
    #Copy code
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    #Install the required packages:
    #Copy code
    pip install -r requirements.txt

##  **🚀 Usage**
Prepare the Data: Ensure that the following data files are available in the specified directories:

Data/Training.csv
Data/Testing.csv
MasterData/symptom_severity.csv
MasterData/symptom_Description.csv
MasterData/symptom_precaution.csv
bg3.jpg (Background image for the GUI)

## Run the Application:

    ```bash
    #Copy code
    python chat_bot.py

## Interact with the GUI:

Enter symptoms in the provided fields.
Click on "DecisionTree" or "SVM" to get predictions.
The results will display possible diseases along with their probabilities and precautionary measures.
