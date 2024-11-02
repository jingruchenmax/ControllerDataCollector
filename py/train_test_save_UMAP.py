import os
import pandas as pd
import umap.umap_ as umap  # Correct import for UMAP
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Function to load and label data from a folder
def load_data_from_folder(folder_path, label):
    data_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            
            # Keep only relevant columns for the right controller
            right_controller_data = data[[
                'right_controller_pos_x', 'right_controller_pos_y', 'right_controller_pos_z',
                'right_controller_rot_w', 'right_controller_rot_x', 'right_controller_rot_y', 'right_controller_rot_z'
            ]].copy()  # Create a copy to avoid SettingWithCopyWarning
            
            # Add label column using .loc
            right_controller_data.loc[:, 'label'] = label
            
            data_list.append(right_controller_data)
    
    return pd.concat(data_list, ignore_index=True)

# Load data from each folder and label them
r_phi_data = load_data_from_folder('preliminary_study_r/r_phi', 'r_phi')
r_square_data = load_data_from_folder('preliminary_study_r/r_square', 'r_square')
r_triangle_data = load_data_from_folder('preliminary_study_r/r_triangle', 'r_triangle')

# Combine all data
all_data = pd.concat([r_phi_data, r_square_data, r_triangle_data], ignore_index=True)

# Separate features and labels
X = all_data.drop(columns=['label'])
y = all_data['label']

# Split data into training and testing sets (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply UMAP for dimensionality reduction
umap_reducer = umap.UMAP(n_components=5, random_state=42)
X_train_umap = umap_reducer.fit_transform(X_train)
X_test_umap = umap_reducer.transform(X_test)

# Train RandomForest Classifier with parallel processing
clf_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)  # Use all available cores
clf_rf.fit(X_train_umap, y_train)

# Train SVM Classifier (SVM by default does not support n_jobs)
clf_svm = SVC(kernel='rbf', C=1, gamma='scale')  # Removed random_state for parallelism if applicable
clf_svm.fit(X_train_umap, y_train)


# Save the models and UMAP reducer
joblib.dump(umap_reducer, 'umap_reducer_umap.pkl')
joblib.dump(clf_rf, 'random_forest_model_umap.pkl')
joblib.dump(clf_svm, 'svm_model_umap.pkl')


# Evaluate RandomForest Classifier
rf_predictions_train = clf_rf.predict(X_train_umap)
rf_predictions_test = clf_rf.predict(X_test_umap)

print("Random Forest Evaluation:")
print("Training Set:")
print(classification_report(y_train, rf_predictions_train))
print("Confusion Matrix (Training):")
print(confusion_matrix(y_train, rf_predictions_train))

print("\nTesting Set:")
print(classification_report(y_test, rf_predictions_test))
print("Confusion Matrix (Testing):")
print(confusion_matrix(y_test, rf_predictions_test))

# Evaluate SVM Classifier
svm_predictions_train = clf_svm.predict(X_train_umap)
svm_predictions_test = clf_svm.predict(X_test_umap)

print("\nSVM Evaluation:")
print("Training Set:")
print(classification_report(y_train, svm_predictions_train))
print("Confusion Matrix (Training):")
print(confusion_matrix(y_train, svm_predictions_train))

print("\nTesting Set:")
print(classification_report(y_test, svm_predictions_test))
print("Confusion Matrix (Testing):")
print(confusion_matrix(y_test, svm_predictions_test))

# Additional evaluation metrics
rf_accuracy_train = accuracy_score(y_train, rf_predictions_train)
rf_accuracy_test = accuracy_score(y_test, rf_predictions_test)

svm_accuracy_train = accuracy_score(y_train, svm_predictions_train)
svm_accuracy_test = accuracy_score(y_test, svm_predictions_test)

print(f"\nRandom Forest Accuracy (Training): {rf_accuracy_train:.2f}")
print(f"Random Forest Accuracy (Testing): {rf_accuracy_test:.2f}")
print(f"SVM Accuracy (Training): {svm_accuracy_train:.2f}")
print(f"SVM Accuracy (Testing): {svm_accuracy_test:.2f}")