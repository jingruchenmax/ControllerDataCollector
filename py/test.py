import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

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
            ]].copy()  # Make a copy to avoid SettingWithCopyWarning
            
            # Add a label column with the folder's label
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

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train and evaluate RandomForest Classifier
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train_pca, y_train)

# Cross-validation for RandomForest
cv_predictions_rf = cross_val_predict(clf_rf, X_train_pca, y_train, cv=5)
print("RandomForest Classification Report for Cross-Validation:")
print(classification_report(y_train, cv_predictions_rf))

# Evaluate RandomForest on the test set
y_pred_rf = clf_rf.predict(X_test_pca)
print("RandomForest Classification Report for Test Set:")
print(classification_report(y_test, y_pred_rf))
print("RandomForest Confusion Matrix for Test Set:")
print(confusion_matrix(y_test, y_pred_rf))

# Train and evaluate Support Vector Machine Classifier
clf_svm = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
clf_svm.fit(X_train_pca, y_train)

# Cross-validation for SVM
cv_predictions_svm = cross_val_predict(clf_svm, X_train_pca, y_train, cv=5)
print("SVM Classification Report for Cross-Validation:")
print(classification_report(y_train, cv_predictions_svm))

# Evaluate SVM on the test set
y_pred_svm = clf_svm.predict(X_test_pca)
print("SVM Classification Report for Test Set:")
print(classification_report(y_test, y_pred_svm))
print("SVM Confusion Matrix for Test Set:")
print(confusion_matrix(y_test, y_pred_svm))
