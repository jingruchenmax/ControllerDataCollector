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
from mpl_toolkits.mplot3d import Axes3D
import datetime
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import MinMaxScaler

# State mapping for numerical labels to string labels
state_mapping = {
    0: 'phi',
    1: 'square',
    2: 'triangle'
}

# Inverse mapping for label conversion
label_map = {'r_phi': 0, 'r_square': 1, 'r_triangle': 2}
inverse_state_mapping = {v: k for k, v in label_map.items()}

# Function to calculate relative position
def calculate_relative_position(hmd_pos, hmd_rot, controller_pos):
    # Convert rotation quaternion (x, y, z, w) to rotation matrix
    rotation = R.from_quat([hmd_rot[0], hmd_rot[1], hmd_rot[2], hmd_rot[3]])
    rotation_matrix = rotation.as_matrix()
    
    # Transform the controller position into the HMD coordinate system
    relative_pos = np.dot(rotation_matrix.T, (controller_pos - hmd_pos).T).T
    return relative_pos

def resample_data(df, target_length=10):
    num_points = len(df)
    if target_length <= num_points:
        # Directly select points at calculated intervals
        indices = np.linspace(0, num_points - 1, target_length, dtype=int)
        resampled_df = df.iloc[indices].reset_index(drop=True)
    else:
        # Pad with zeros equally at the start and end
        padding_length = target_length - num_points
        pad_start = padding_length // 2
        pad_end = padding_length - pad_start
        
        # Create a DataFrame with zeros for padding
        zero_padding_start = pd.DataFrame(0, index=range(pad_start), columns=df.columns)
        zero_padding_end = pd.DataFrame(0, index=range(pad_end), columns=df.columns)
        
        # Concatenate zero padding and the original DataFrame
        resampled_df = pd.concat([zero_padding_start, df, zero_padding_end], ignore_index=True)
    # Prepare the resampled data for rendering (keep only relevant columns)

    return resampled_df



def process_data(base_dir, label_map, target_length=10):
    processed_data = []  # For full 70-dimensional data for model training
    original_3d_data = {label: [] for label in label_map.values()}  # Collect original 3D data by label
    resampled_3d_data = {label: [] for label in label_map.values()}  # Collect resampled 3D data by label

    cols_to_extract = ['right_controller_pos_x', 'right_controller_pos_y', 'right_controller_pos_z',
                       'right_controller_rot_w', 'right_controller_rot_x', 'right_controller_rot_y', 'right_controller_rot_z',
                       'hmd_pos_x', 'hmd_pos_y', 'hmd_pos_z', 'hmd_rot_w', 'hmd_rot_x', 'hmd_rot_y', 'hmd_rot_z']
    original_3d_columns = ['relative_right_pos_x', 'relative_right_pos_y', 'relative_right_pos_z']

    scaler = MinMaxScaler(feature_range=(0, 1))  # Create a MinMaxScaler instance
    for label_name, label in label_map.items():
        folder_path = os.path.join(base_dir, label_name)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist")
            continue
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(folder_path, filename))

                # Ensure all required columns are present
                required_cols = cols_to_extract
                if not all(col in df.columns for col in required_cols):
                    print(f"Warning: Missing required columns in {filename}")
                    continue

                # Extract HMD and controller positions and rotations
                hmd_pos = df[['hmd_pos_x', 'hmd_pos_y', 'hmd_pos_z']].values
                hmd_rot = df[['hmd_rot_x', 'hmd_rot_y', 'hmd_rot_z', 'hmd_rot_w']].values
                controller_pos = df[['right_controller_pos_x', 'right_controller_pos_y', 'right_controller_pos_z']].values

                # Calculate relative position for each time step
                try:
                    relative_positions = np.array([
                        calculate_relative_position(hmd_pos[i], hmd_rot[i], controller_pos[i])
                        for i in range(len(df))
                    ])
                except Exception as e:
                    print(f"Error calculating relative positions in {filename}: {e}")
                    print("HMD Position:", hmd_pos[i])
                    print("HMD Rotation (quaternion):", hmd_rot[i])
                    print("Controller Position:", controller_pos[i])
                    continue
                
                # Scale the relative position data to [0, 1] range
                scaled_relative_positions = scaler.fit_transform(relative_positions)

                # Add relative positions to the DataFrame
                df['relative_right_pos_x'] = scaled_relative_positions[:, 0]
                df['relative_right_pos_y'] = scaled_relative_positions[:, 1]
                df['relative_right_pos_z'] = scaled_relative_positions[:, 2]
                
                # List of substrings that indicate HMD and left controller data columns
                columns_to_drop = [col for col in df.columns if 'hmd' in col or 'left_controller' in col or 'timestamp' in col or 'right_controller_pos' in col ]

                # Create a new DataFrame without HMD and left controller columns
                updated_df = df.drop(columns=columns_to_drop)

                # Collect relative positions for plotting
                relative_3d_df = df[original_3d_columns].copy()
                relative_3d_df['label'] = label  # Add label for plotting purposes
                original_3d_data[label].append(relative_3d_df)

                # Resample the relative positions for training
                # Pass only the numeric columns for resampling
                resampled_features = resample_data(updated_df, target_length=target_length)
                # Add label to the resampled DataFrame for consistency
                resampled_3d_df = resampled_features[['relative_right_pos_x', 'relative_right_pos_y', 'relative_right_pos_z']].copy()
                resampled_3d_df['label'] = label
                resampled_3d_data[label].append(resampled_3d_df)

                # Flatten resampled features into one row for training
                flattened_features = {}
                for col in resampled_features:
                    for i in range(target_length):
                        flattened_features[f"{col}_{i}"] = resampled_features[col][i]

                flattened_features['label'] = label
                processed_data.append(flattened_features)

    # Combine processed data for training into a single DataFrame
    processed_data_df = pd.DataFrame(processed_data)
    return processed_data_df, original_3d_data, resampled_3d_data



def plot_3d_comparison(original_data_by_label, resampled_data_by_label, original_3d_columns=['x', 'y', 'z'], state_mapping=state_mapping):
    for label, original_data_list in original_data_by_label.items():
        label_name = state_mapping.get(label, f"Label {label}")

        fig = plt.figure(figsize=(15, 7))

        # Plot original data
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title(f'Original 3D Points for {label_name}')
        ax1.set_xlabel(original_3d_columns[0])
        ax1.set_ylabel(original_3d_columns[1])
        ax1.set_zlabel(original_3d_columns[2])
        
        # Plot all original data chunks
        for original_df in original_data_list:
            if isinstance(original_df, pd.DataFrame):  # Ensure original_df is a DataFrame
                ax1.plot(
                    original_df[original_3d_columns[0]], 
                    original_df[original_3d_columns[1]], 
                    original_df[original_3d_columns[2]], 
                    color='blue', marker='o', linewidth=1, alpha=0.7
                )

        # Add one legend for the original data
        ax1.plot([], [], 'bo', label='Original Data')
        ax1.legend()

        # Plot resampled data
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title(f'Resampled 3D Points for {label_name}')
        ax2.set_xlabel(original_3d_columns[0])
        ax2.set_ylabel(original_3d_columns[1])
        ax2.set_zlabel(original_3d_columns[2])

        for resampled_df in resampled_data_by_label[label]:
            if isinstance(resampled_df, pd.DataFrame):  # Ensure resampled_df is a DataFrame
                ax2.plot(
                    resampled_df['relative_right_pos_x'], 
                    resampled_df['relative_right_pos_y'], 
                    resampled_df['relative_right_pos_z'], 
                    color='red', marker='^', linewidth=1, alpha=0.7
                )
            else:
                print(f"Warning: Skipping non-DataFrame object in resampled data for label {label_name}")

        # Add one legend for the resampled data
        ax2.plot([], [], 'r^', label='Resampled Data')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"3D_Comparison_{label_name}_Relative.png")
        plt.show()




def plot_3d_examples_by_label(original_data_by_label, resampled_data_by_label, original_3d_columns=['x', 'y', 'z'], state_mapping=state_mapping):
    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.set_title('Original 3D Points (One Example per Label)')
    ax1.set_xlabel(original_3d_columns[0])
    ax1.set_ylabel(original_3d_columns[1])
    ax1.set_zlabel(original_3d_columns[2])

    ax2.set_title('Resampled 3D Points (One Example per Label)')
    ax2.set_xlabel(original_3d_columns[0])
    ax2.set_ylabel(original_3d_columns[1])
    ax2.set_zlabel(original_3d_columns[2])

    colors = ['blue', 'green', 'red']
    
    for i, (label, original_data_list) in enumerate(original_data_by_label.items()):
        label_name = state_mapping.get(label, f"Label {label}")
        color = colors[i % len(colors)]

        # Plot one example from the original data
        if original_data_list:
            original_example = original_data_list[0]  # First time series chunk
            ax1.plot(
                original_example[original_3d_columns[0]], 
                original_example[original_3d_columns[1]], 
                original_example[original_3d_columns[2]], 
                label=f'Original {label_name}', color=color, marker='o', linewidth=1, alpha=1
            )

        # Plot one example from the resampled data
        if resampled_data_by_label[label]:
            resampled_example = resampled_data_by_label[label][0]  # First resampled chunk
            ax2.plot(
                resampled_example['relative_right_pos_x'], 
                resampled_example['relative_right_pos_y'], 
                resampled_example['relative_right_pos_z'], 
                label=f'Resampled {label_name}', color=color, marker='^', linewidth=1, alpha=1
            )

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig("3D_Example_Comparison_Relative.png")
    plt.show()

# Function to plot 2D comparison of original and resampled data
def plot_2d_comparison(original_data_by_label, resampled_data_by_label, state_mapping=state_mapping):
    for label, original_data_list in original_data_by_label.items():
        label_name = state_mapping.get(label, f"Label {label}")

        plt.figure(figsize=(14, 6))

        # Plot original data
        plt.subplot(1, 2, 1)
        plt.title(f'Original 2D Points for {label_name}')
        plt.xlabel('X')
        plt.ylabel('Y')

        # Plot all original data chunks
        for original_df in original_data_list:
            if isinstance(original_df, pd.DataFrame):  # Ensure original_df is a DataFrame
                plt.plot(
                    original_df['relative_right_pos_x'],
                    original_df['relative_right_pos_y'],
                    color='blue', marker='o', linewidth=1, alpha=0.7
                )

        plt.plot([], [], 'bo', label='Original Data')
        plt.legend()

        # Plot resampled data
        plt.subplot(1, 2, 2)
        plt.title(f'Resampled 2D Points for {label_name}')
        plt.xlabel('X')
        plt.ylabel('Y')

        for resampled_df in resampled_data_by_label[label]:
            if isinstance(resampled_df, pd.DataFrame):  # Ensure resampled_df is a DataFrame
                plt.plot(
                    resampled_df['relative_right_pos_x'],
                    resampled_df['relative_right_pos_y'],
                    color='red', marker='^', linewidth=1, alpha=0.7
                )

        plt.plot([], [], 'r^', label='Resampled Data')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"2D_Comparison_{label_name}_Relative.png")
        plt.show()

# Function for training and evaluating models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Cross-validation setup only on the training set
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Train and evaluate Random Forest
    rf_classifier = RandomForestClassifier(n_estimators=30, # Reduce number of trees if overfitting
                                           random_state=42,    
                                           max_depth=10,      # Limit tree depth
                                           min_samples_split=6,  # Require more samples to split nodes
                                           min_samples_leaf=4)   # Require more samples at leaf nodes
    rf_classifier.fit(X_train, y_train)
    rf_scores = cross_val_score(rf_classifier, X_train, y_train, cv=cv, scoring='accuracy')
    print("\nRandom Forest Cross-Validation Results on Training Set:")
    print(f"Mean Accuracy: {np.mean(rf_scores):.2f} ± {np.std(rf_scores):.2f}")

    print("\nEvaluating Random Forest on Test Set:")
    evaluate_model(rf_classifier, X_test, y_test, "Random Forest")

    # Train and evaluate SVM
    svm_classifier = SVC(
                        kernel='rbf',
                        probability=True,
                        gamma=0.5,
                        C=0.1,  # Increase regularization strength
                        random_state=42)
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

# Function to evaluate a model and save results to a .txt file with the time as the filename
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_named = [inverse_state_mapping[pred] for pred in y_pred]
    y_test_named = [inverse_state_mapping[true] for true in y_test]

    accuracy = accuracy_score(y_test_named, y_pred_named)
    precision = precision_score(y_test_named, y_pred_named, average='weighted')
    recall = recall_score(y_test_named, y_pred_named, average='weighted')
    f1 = f1_score(y_test_named, y_pred_named, average='weighted')
    report = classification_report(y_test_named, y_pred_named)

    # Print results to the console
    print(f"\n{model_name} Model Test Evaluation:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print("\nClassification Report:\n", report)

    # Ensure only labels that exist in y_test_named are passed to the confusion matrix
    unique_labels = sorted(set(y_test_named))
    conf_matrix = confusion_matrix(y_test_named, y_pred_named, labels=unique_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap((conf_matrix / conf_matrix.sum(axis=1, keepdims=True)), annot=True, fmt='.2%', cmap='Blues' if model_name == "Random Forest" else 'Oranges',
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"Confusion_Matrix_{model_name}.png")
    plt.show()

    # Save results to a text file with the current timestamp as the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_evaluation_{timestamp}.txt"
    
    with open(filename, 'w') as file:
        file.write(f"{model_name} Model Test Evaluation:\n")
        file.write(f"Accuracy: {accuracy:.2f}\n")
        file.write(f"Precision: {precision:.2f}\n")
        file.write(f"Recall: {recall:.2f}\n")
        file.write(f"F1-Score: {f1:.2f}\n")
        file.write("\nClassification Report:\n")
        file.write(report)
        file.write("\nConfusion Matrix:\n")
        file.write(np.array2string(conf_matrix))

    print(f"Results saved to {filename}")

base_dir = 'preliminary_study_r'
target_length = 10  # Change this to adjust the length of resampling

# Process data to get all three outputs
processed_data_df, original_3d_data_by_label, resampled_3d_data_by_label = process_data(base_dir, label_map, target_length=target_length)

print(processed_data_df)
# plot_3d_comparison(
#     original_data_by_label=original_3d_data_by_label, 
#     resampled_data_by_label=resampled_3d_data_by_label, 
#     original_3d_columns=['relative_right_pos_x', 'relative_right_pos_y', 'relative_right_pos_z']
# )

# # Call plot_3d_examples_by_label to plot one sample from each label group
# plot_3d_examples_by_label(
#     original_data_by_label=original_3d_data_by_label, 
#     resampled_data_by_label=resampled_3d_data_by_label, 
#     original_3d_columns=['relative_right_pos_x', 'relative_right_pos_y', 'relative_right_pos_z']
# )

# plot_2d_comparison(
#     original_data_by_label=original_3d_data_by_label,
#     resampled_data_by_label=resampled_3d_data_by_label,
#     state_mapping=state_mapping
# )

# Separate features and labels for training
X = processed_data_df.drop(columns=['label'])
y = processed_data_df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Ensure that training and testing sets are properly separated
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

train_and_evaluate_models(X_train, X_test, y_train, y_test)
