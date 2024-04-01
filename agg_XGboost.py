#from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


#data initialization
fileToUse = 'avg_top10_augmented.csv'    #csvs generated in Feature Engineering And Bootstrapping.py
df = pd.read_csv(fileToUse)

#splitting
y = df['attacked']  # Target variable
X = df.drop('attacked', axis=1)  # Drop target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)    #random_state=42 was the seed I originally tested on
#smote = SMOTE(random_state=42)
#X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
# Fit the scaler to the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

params = {      #parameters for xgboost classifier
    'device': 'cuda',
    'learning_rate': 0.2,
    'n_estimators': 1000,
    'scale_pos_weight': 0.2,
    'reg_alpha': 60,    #I think 45 works best for bootstrapping so far, represents L1 Regularization
    'reg_lambda': 0,

}
xgb_classifier = xgb.XGBClassifier(**params)

def plot_feature_importance(x_train, y_train, model):
    model.fit(x_train,y_train)
    # Get feature importance
    feature_importance = xgb_classifier.feature_importances_

    # Get feature names
    feature_names = X_train.columns

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importance)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.show()
def plot_learning_curves(X_train, y_train, X_test, y_test, model):
    train_errors, test_errors = [], []
    sizes = np.arange(0.1, 1.1, 0.1)  # Training dataset sizes

    for size in sizes:
        # Train the model on a subset of the training data
        subset_size = int(size * len(X_train))
        X_subset, y_subset = X_train[:subset_size], y_train[:subset_size]
        model.fit(X_subset, y_subset)

        # Make predictions on the training and test sets
        train_predictions = model.predict(X_subset)
        test_predictions = model.predict(X_test)

        # Calculate the accuracy scores
        train_accuracy = accuracy_score(y_subset, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)

        # Append the accuracy scores to the lists
        train_errors.append(train_accuracy)
        test_errors.append(test_accuracy)

    # Plot the learning curves
    plt.plot(sizes, train_errors, label='Training Set')
    plt.plot(sizes, test_errors, label='Test Set')
    plt.title('Learning Curves')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def printfinalMetrics(X_train, y_train, X_test, y_test, model):
    # Assuming X_train, y_train are your training features and labels
    # Assuming X_test, y_test are your test features and labels
    # Assuming xgb_classifier is your trained XGBoost classifier

    # Train the classifier
    model.fit(X_train, y_train)

    # Make train predictions
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_precision = precision_score(y_train, train_predictions)
    train_recall = recall_score(y_train, train_predictions)
    train_f1 = f1_score(y_train, train_predictions)

    # Make predictions
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(y_test, test_predictions)
    test_recall = recall_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions)


    print("Train Accuracy:", train_accuracy)
    print("Train precision:", train_precision)
    print("Train recall:", train_recall)
    print("Train f1 score:", train_f1)

    print("Test Accuracy:", test_accuracy)
    print("test precision:", test_precision)
    print("test recall:", test_recall)
    print("test f1 score:", test_f1)

plot_learning_curves(X_train_scaled, y_train, X_test_scaled, y_test, xgb_classifier)
printfinalMetrics(X_train_scaled, y_train, X_test_scaled, y_test, xgb_classifier)
#plot_feature_importance(X_train_scaled, y_train, xgb_classifier)