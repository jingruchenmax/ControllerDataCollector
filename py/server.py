from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
import numpy as np
from scipy.signal import welch
import traceback

# Load pre-trained models
try:
    clf_rf = joblib.load('random_forest_classifier_with_labels.pkl')
    clf_svm = joblib.load('svm_classifier_with_labels.pkl')
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

app = Flask(__name__)

# State mapping for numerical labels to string labels
state_mapping = {
    0: 'phi',
    1: 'square',
    2: 'triangle'
}

# Function to compute spectral power for a time series and return the array
def compute_spectral_power(series, fs=1.0):
    try:
        nperseg = min(256, len(series))
        freqs, power = welch(series, fs=fs, nperseg=nperseg)
        log_power = np.log1p(power)  # Log transformation for normalization
        return log_power
    except Exception as e:
        print(f"Error in compute_spectral_power for series {series.name}: {e}")
        print(traceback.format_exc())
        raise

# Function to extract features and flatten arrays into separate columns
def extract_features_from_chunk(df):
    try:
        features = {}
        fs = 1.0  # Sampling frequency, adjust as needed
        
        # Extract spectral power for relevant columns
        for col in ['right_controller_pos_x', 'right_controller_pos_y', 'right_controller_pos_z',
                    'right_controller_rot_w', 'right_controller_rot_x', 'right_controller_rot_y', 'right_controller_rot_z']:
            if col in df:
                print(f"Processing column: {col}")
                spectral_power = compute_spectral_power(df[col], fs)
                print(f"Spectral power for {col}: {spectral_power[:5]}")
                for i, value in enumerate(spectral_power):
                    features[f'spectral_power_{col}'] = value
            else:
                print(f"Warning: {col} not found in DataFrame")
        
        return pd.DataFrame([features])
    except Exception as e:
        print(f"Error in extract_features_from_chunk: {e}")
        print(traceback.format_exc())
        raise

@app.route('/classify', methods=['POST'])
def classify():
    try:
        print("Received a request...")
        raw_data = request.data.decode('utf-8')
        print("Raw data decoded.")

        data = json.loads(raw_data)
        print("JSON data parsed successfully.")

        if 'data' not in data or not isinstance(data['data'], list):
            print("Invalid data format: 'data' key not found or not a list.")
            return jsonify({'error': 'Invalid data format'}), 400

        df = pd.DataFrame(data['data'])
        print(f"Data converted to DataFrame with shape: {df.shape}")
        print(f"DataFrame head:\n{df.head()}")

        features_df = extract_features_from_chunk(df)
        print("Features extracted successfully.")
        print(f"Extracted features DataFrame shape: {features_df.shape}")
        print(f"Extracted features DataFrame head:\n{features_df.head()}")

        rf_prediction = clf_rf.predict(features_df)[0]
        svm_prediction = clf_svm.predict(features_df)[0]

        rf_prediction_label = state_mapping.get(rf_prediction, 'Unknown')
        svm_prediction_label = state_mapping.get(svm_prediction, 'Unknown')

        rf_confidence = None
        svm_confidence = None

        if hasattr(clf_rf, 'predict_proba'):
            rf_probabilities = clf_rf.predict_proba(features_df)
            rf_confidence = np.max(rf_probabilities)
            print(f"Random Forest probabilities: {rf_probabilities}")
            print(f"Random Forest confidence: {rf_confidence:.2f}")

        if hasattr(clf_svm, 'predict_proba'):
            svm_probabilities = clf_svm.predict_proba(features_df)
            svm_confidence = np.max(svm_probabilities)
            print(f"SVM probabilities: {svm_probabilities}")
            print(f"SVM confidence: {svm_confidence:.2f}")

        print(f"Random Forest prediction: {rf_prediction_label} (Confidence: {rf_confidence:.2f})" if rf_confidence is not None else f"Random Forest prediction: {rf_prediction_label} (Confidence: N/A)")
        print(f"SVM prediction: {svm_prediction_label} (Confidence: {svm_confidence:.2f})" if svm_confidence is not None else f"SVM prediction: {svm_prediction_label} (Confidence: N/A)")

        response = {
            'rf_result': {
                'predicted_class': rf_prediction_label,
                'confidence_score': round(rf_confidence, 2) if rf_confidence is not None else 'N/A'
            },
            'svm_result': {
                'predicted_class': svm_prediction_label,
                'confidence_score': round(svm_confidence, 2) if svm_confidence is not None else 'N/A'
            }
        }
        return jsonify(response)

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return jsonify({'error': f'JSON parsing error: {str(e)}'}), 400
    except KeyError as e:
        print(f"Key error: {e}")
        return jsonify({'error': f'Missing key in data: {str(e)}'}), 400
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
