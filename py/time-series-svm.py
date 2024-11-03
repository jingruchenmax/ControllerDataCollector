import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# Function to load the dataset
def load_time_series_data(base_dir):
    data = []
    labels = []
    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(subfolder_path, file)
                    df = pd.read_csv(file_path)
                    if {'right_controller_pos_x', 'right_controller_pos_y', 'right_controller_pos_z'}.issubset(df.columns):
                        series_data = pd.DataFrame({
                            'x': pd.Series(df['right_controller_pos_x'].values),
                            'y': pd.Series(df['right_controller_pos_y'].values),
                            'z': pd.Series(df['right_controller_pos_z'].values)
                        })
                        data.append(series_data)
                        labels.append(subfolder)
    return pd.DataFrame({'dim_0': [d['x'] for d in data], 
                         'dim_1': [d['y'] for d in data], 
                         'dim_2': [d['z'] for d in data]}), pd.Series(labels)

# Function to resample time series data to a unified length
def resample_time_series_data(X, target_length=100, method='interpolate'):
    """
    Resample the time series data in each cell to a unified length.
    
    Parameters:
    - X: pd.DataFrame with each cell containing a pd.Series representing time series data.
    - target_length: int, the length to which all series should be resampled.
    - method: str, 'interpolate' for interpolation.
    
    Returns:
    - resampled_data: pd.DataFrame with resampled time series data.
    """
    X_resampled = pd.DataFrame()
    
    for col in X.columns:
        resampled_col = []
        
        for series in X[col]:
            if method == 'interpolate':
                # Create a new index to interpolate to
                new_index = np.linspace(0, len(series) - 1, target_length)
                interpolated_series = pd.Series(np.interp(new_index, np.arange(len(series)), series))
                resampled_col.append(interpolated_series)
        
        X_resampled[col] = resampled_col
    
    return X_resampled

# Load and resample the dataset
base_directory_path = 'preliminary_study_r'  # Update this path with your dataset location
X, y = load_time_series_data(base_directory_path)
X_resampled = resample_time_series_data(X, target_length=10, method='interpolate')

# Transform resampled data into a 300-dimensional input format for SVM
X_flattened = pd.DataFrame({
    'flattened_series': X_resampled.apply(
        lambda row: np.concatenate([row['dim_0'], row['dim_1'],row['dim_2']]), axis=1
    )
})

X_flattened = pd.DataFrame(X_flattened['flattened_series'].tolist(), index=X_flattened.index)
X_flattened['label'] = y  # Add labels for export

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_flattened.drop(columns='label'), y, test_size=0.5, random_state=42)

# Save the training and testing data to CSV files
train_set = X_train.copy()
train_set['label'] = y_train
train_set.to_csv('train_set.csv', index=False)

test_set = X_test.copy()
test_set['label'] = y_test
test_set.to_csv('test_set.csv', index=False)

print("Training and testing sets saved as 'train_set.csv' and 'test_set.csv'.")

# Initialize and train the SVM classifier
svm_clf = SVC(kernel='rbf', C=1.0)
print("Training the SVM classifier...")
svm_clf.fit(X_train, y_train)

# Save the trained model
model_filename = 'experiment_SVM_classifier.pkl'
joblib.dump(svm_clf, model_filename)
print(f"Trained SVM model saved as {model_filename}")

# Evaluate the model on the testing set
y_test_pred = svm_clf.predict(X_test)
print("\nTesting Set Evaluation Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred, average='weighted', zero_division=0):.4f}")
print(f"F1 Score: {f1_score(y_test, y_test_pred, average='weighted', zero_division=0):.4f}")

# Print classification report for the testing set
print("\nClassification Report for Testing Set:")
print(classification_report(y_test, y_test_pred, zero_division=0))

# Evaluate the model on the training set
y_train_pred = svm_clf.predict(X_train)
print("\nTraining Set Evaluation Metrics:")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Precision: {precision_score(y_train, y_train_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_train, y_train_pred, average='weighted', zero_division=0):.4f}")
print(f"F1 Score: {f1_score(y_train, y_train_pred, average='weighted', zero_division=0):.4f}")

# Print classification report for the training set
print("\nClassification Report for Training Set:")
print(classification_report(y_train, y_train_pred, zero_division=0))
