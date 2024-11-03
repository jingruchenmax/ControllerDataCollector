import os
import pandas as pd
import numpy as np
from scipy.signal import welch
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# State mapping for numerical labels to string labels
state_mapping = {
    0: 'phi',
    1: 'square',
    2: 'triangle'
}

# Inverse mapping for label conversion
label_map = {'r_phi': 0, 'r_square': 1, 'r_triangle': 2}
inverse_state_mapping = {v: k for k, v in label_map.items()}

# Function to compute spectral power for a time series
def compute_spectral_power(series, fs=1.0):
    nperseg = min(256, len(series))
    freqs, power = welch(series, fs=fs, nperseg=nperseg)
    return np.log1p(power)

# Load and process data
def process_data(base_dir, label_map):
    processed_data = []
    cols_to_extract = ['right_controller_pos_x', 'right_controller_pos_y', 'right_controller_pos_z',
                       'right_controller_rot_w', 'right_controller_rot_x', 'right_controller_rot_y', 'right_controller_rot_z']

    for label_name, label in label_map.items():
        folder_path = os.path.join(base_dir, label_name)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist")
            continue
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(folder_path, filename))
                df_filtered = df[cols_to_extract]
                features = {f'spectral_power_{col}': compute_spectral_power(df_filtered[col]) for col in cols_to_extract}
                features['label'] = label  # Include the numerical label
                processed_data.append(pd.DataFrame([features]))

    return pd.concat(processed_data, ignore_index=True)

# Main code execution
base_dir = 'preliminary_study_r'
df = process_data(base_dir, label_map)

# Flatten the data to only keep spectral feature columns and the label
flattened_data = []
for _, row in df.iterrows():
    flattened = {key: np.mean(row[key]) for key in row.index if key.startswith('spectral_power_')}
    flattened['label'] = row['label']
    flattened_data.append(flattened)

flattened_df = pd.DataFrame(flattened_data)

# Separate features and labels
X = flattened_df.drop(columns=['label'])
y = flattened_df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Ensure that training and testing sets are properly separated
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Function for training and evaluating models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Cross-validation setup only on the training set
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=52)

    # Train and evaluate Random Forest
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    rf_scores = cross_val_score(rf_classifier, X_train, y_train, cv=cv, scoring='accuracy')
    print("\nRandom Forest Cross-Validation Results on Training Set:")
    print(f"Mean Accuracy: {np.mean(rf_scores):.2f} ± {np.std(rf_scores):.2f}")

    print("\nEvaluating Random Forest on Test Set:")
    evaluate_model(rf_classifier, X_test, y_test, "Random Forest")

    # Train and evaluate SVM
    svm_classifier = SVC(kernel='rbf', random_state=42, probability=True)
    svm_classifier.fit(X_train, y_train)
    svm_scores = cross_val_score(svm_classifier, X_train, y_train, cv=cv, scoring='accuracy')
    print("\nSVM Cross-Validation Results on Training Set:")
    print(f"Mean Accuracy: {np.mean(svm_scores):.2f} ± {np.std(svm_scores):.2f}")

    print("\nEvaluating SVM on Test Set:")
    evaluate_model(svm_classifier, X_test, y_test, "SVM")

    # Save models
    joblib.dump(rf_classifier, 'random_forest_classifier_with_labels.pkl')
    joblib.dump(svm_classifier, 'svm_classifier_with_labels.pkl')
    print("\nModels saved successfully.")

# Function to evaluate a model
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_named = [inverse_state_mapping[pred] for pred in y_pred]
    y_test_named = [inverse_state_mapping[true] for true in y_test]

    print(f"\n{model_name} Model Test Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test_named, y_pred_named):.2f}")
    print(f"Precision: {precision_score(y_test_named, y_pred_named, average='weighted'):.2f}")
    print(f"Recall: {recall_score(y_test_named, y_pred_named, average='weighted'):.2f}")
    print(f"F1-Score: {f1_score(y_test_named, y_pred_named, average='weighted'):.2f}")
    print("\nClassification Report:\n", classification_report(y_test_named, y_pred_named))

    # Ensure only labels that exist in y_test_named are passed to the confusion matrix
    unique_labels = sorted(set(y_test_named))
    conf_matrix = confusion_matrix(y_test_named, y_pred_named, labels=unique_labels)

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues' if model_name == "Random Forest" else 'Oranges',
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

train_and_evaluate_models(X_train, X_test, y_train, y_test)
