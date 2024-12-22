# -*- coding: utf-8 -*-
"""
Spyder Editor
Created on fri Dec 20 03:03:11 2024

@author: hugod
"""

import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

TEST_SIZE = 0.4

# Declare 'filename' as a global variable to be used by default if no command-line argument is given.
filename = 'shopping.csv'

def main():
    global filename  # This tells Python that we're using the global variable 'filename'
    
    # Check command-line arguments
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        print("No command line argument provided for data. Using default:", filename)

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(filename)
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def load_data(filename):
    """
    Load shopping data from a CSV file and convert into lists of evidence and labels.
    """
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header

        evidence = []
        labels = []
        month_map = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5, 'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}

        for row in reader:
            # Convert columns to appropriate data types
            row_data = [
                int(row[0]),  # Administrative
                float(row[1]),  # Administrative_Duration
                int(row[2]),  # Informational
                float(row[3]),  # Informational_Duration
                int(row[4]),  # ProductRelated
                float(row[5]),  # ProductRelated_Duration
                float(row[6]),  # BounceRates
                float(row[7]),  # ExitRates
                float(row[8]),  # PageValues
                float(row[9]),  # SpecialDay
                month_map[row[10]],  # Month
                int(row[11]),  # OperatingSystems
                int(row[12]),  # Browser
                int(row[13]),  # Region
                int(row[14]),  # TrafficType
                1 if row[15] == 'Returning_Visitor' else 0,  # VisitorType
                1 if row[16] == 'TRUE' else 0  # Weekend
            ]
            evidence.append(row_data)
            labels.append(1 if row[17] == 'TRUE' else 0)

    return evidence, labels

def train_model(evidence, labels):
    """
    Train a k-nearest neighbor model (k=1).
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
    """
    Evaluate the model to determine sensitivity and specificity.
    """
    cm = confusion_matrix(labels, predictions)
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) != 0 else 0
    specificity = cm[0, 0] / (cm[0, 1] + cm[0, 0]) if (cm[0, 1] + cm[0, 0]) != 0 else 0
    return sensitivity, specificity

if __name__ == "__main__":
    main()
