#Model Deployment and Prediction:
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained model
model = joblib.load('models/trained_model.pkl')

# Function to predict fraud
def predict_fraud(transaction_data):
    prediction = model.predict([transaction_data])
    if prediction == 1:
        return "Fraudulent"
    else:
        return "Non-Fraudulent"

# Command-line interface
def main():
    print("Credit Card Fraud Detection")
    print("--------------------------")

    while True:
        try:
            amount = float(input("Enter transaction amount: "))
            time = float(input("Enter transaction time: "))
            feature1 = float(input("Enter numerical feature 1: "))
            feature2 = float(input("Enter numerical feature 2: "))
            
            transaction_data = [amount, time, feature1, feature2]

            prediction = predict_fraud(transaction_data)
            print(f"The transaction is predicted to be: {prediction}")
            
            another = input("Do you want to predict another transaction? (yes/no): ")
            if another.lower() != 'yes':
                break
        except ValueError:
            print("Invalid input. Please enter numeric values.")

if __name__ == "__main__":
    main()
