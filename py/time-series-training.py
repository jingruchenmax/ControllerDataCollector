import os
import pandas as pd
import numpy as np
from scipy.signal import welch
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, log_loss, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Mapping of state numbers to their names
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

# Function to extract only spectral power features for specific columns in a DataFrame
def extract_features(df, columns, fs=1.0):
    features = {f'spectral_power_{col}': compute_spectral_power(df[col], fs) for col in columns}
    return features

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
                features = extract_features(df_filtered, cols_to_extract)
                features['label'] = label  # Include the numerical label
                processed_data.append(pd.DataFrame([features]))

    # Combine only the columns related to spectral values and label
    return pd.concat(processed_data, ignore_index=True)

# Main code execution
base_dir = 'preliminary_study_r'
df = process_data(base_dir, label_map)

# Apply flattening to keep only spectral feature columns and the label
flattened_data = []
for _, row in df.iterrows():
    flattened = {key: np.mean(row[key]) for key in row.index if key.startswith('spectral_power_')}
    flattened['label'] = row['label']
    flattened_data.append(flattened)

flattened_df = pd.DataFrame(flattened_data)

# Continue with the rest of the script for training and evaluating models
X = flattened_df.drop(columns=['label'])
y = flattened_df['label']

# Function for training and evaluating models
def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50, stratify=y)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=52)

    # Random Forest
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_scores = cross_val_score(rf_classifier, X, y, cv=cv, scoring='accuracy')
    print("\nRandom Forest Cross-Validation Results:")
    print(f"Mean Accuracy: {np.mean(rf_scores):.2f} ± {np.std(rf_scores):.2f}")

    rf_classifier.fit(X_train, y_train)
    evaluate_model(rf_classifier, X_test, y_test, "Random Forest")

    # SVM
    svm_classifier = SVC(kernel='rbf', random_state=42, probability=True)
    svm_scores = cross_val_score(svm_classifier, X, y, cv=cv, scoring='accuracy')
    print("\nSVM Cross-Validation Results:")
    print(f"Mean Accuracy: {np.mean(svm_scores):.2f} ± {np.std(svm_scores):.2f}")

    svm_classifier.fit(X_train, y_train)
    evaluate_model(svm_classifier, X_test, y_test, "SVM")

    # Save models
    joblib.dump(rf_classifier, 'random_forest_classifier_with_labels.pkl')
    joblib.dump(svm_classifier, 'svm_classifier_with_labels.pkl')
    print("\nModels saved successfully with labels embedded.")

# Function to evaluate a model
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_named = [inverse_state_mapping[pred] for pred in y_pred]
    y_test_named = [inverse_state_mapping[true] for true in y_test]

    print(f"\n{model_name} Model Evaluation:")
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

    if hasattr(model, 'predict_proba'):
        y_pred_prob = model.predict_proba(X_test)
        if len(unique_labels) == 2:  # Only for binary classification
            roc_auc = roc_auc_score(y_test, y_pred_prob[:, 1])
            print(f"ROC-AUC Score: {roc_auc:.2f}")
            log_loss_value = log_loss(y_test, y_pred_prob)
            print(f"Log Loss: {log_loss_value:.2f}")


train_and_evaluate_models(X, y)
