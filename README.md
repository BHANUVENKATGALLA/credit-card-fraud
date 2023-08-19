import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix, precision_score, recall_score
import plotly.express as px

# Load the dataset
data = pd.read_csv('c:\creditcard.csv')

# Assuming you have already performed data preprocessing, feature engineering, and defined X and y

# Define X and y based on your dataset
X = data.drop('Class', axis=1)  # Drop the target variable
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
svc = SVC()
nb = GaussianNB()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()

models = [svc, nb, dtc, rfc]

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)  # Set zero_division to 1
    recall = recall_score(y_test, y_pred)
    
    print(type(model).__name__, "Model Test Accuracy Score is:", accuracy)
    print(type(model).__name__, "Model Test F1 Score is:", f1)
    print(type(model).__name__, "Model Test Precision Score is:", precision)
    print(type(model).__name__, "Model Test Recall Score is:", recall)
