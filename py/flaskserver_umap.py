from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
from collections import Counter

# Load pre-trained models
umap_reducer = joblib.load('umap_reducer_umap.pkl')
clf_rf = joblib.load('random_forest_model_umap.pkl')
clf_svm = joblib.load('svm_model_umap.pkl')

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Get raw data and decode it
        raw_data = request.data.decode('utf-8')

        # Parse the JSON data
        data = json.loads(raw_data)

        # Check if the 'data' key exists and contains a list
        if 'data' not in data or not isinstance(data['data'], list):
            return jsonify({'error': 'Invalid data format'}), 400

        # Extract the list of data points
        data_points = data['data']

        # Convert the list of data points to a DataFrame
        df = pd.DataFrame(data_points)

        # Select only the columns of interest
        columns_of_interest = [
            'right_controller_pos_x', 'right_controller_pos_y', 'right_controller_pos_z',
            'right_controller_rot_w', 'right_controller_rot_x', 'right_controller_rot_y', 'right_controller_rot_z'
        ]
        df = df[columns_of_interest]

        # Apply UMAP transformation
        X_umap = umap_reducer.transform(df)

        # Get predictions from both classifiers
        rf_prediction = clf_rf.predict(X_umap)
        svm_prediction = clf_svm.predict(X_umap)

        # Calculate most likely prediction and confidence score for RandomForest
        rf_counter = Counter(rf_prediction)
        rf_most_common = rf_counter.most_common(1)[0]  # (most_common_class, count)
        rf_confidence = rf_most_common[1] / len(rf_prediction)

        # Calculate most likely prediction and confidence score for SVM
        svm_counter = Counter(svm_prediction)
        svm_most_common = svm_counter.most_common(1)[0]  # (most_common_class, count)
        svm_confidence = svm_most_common[1] / len(svm_prediction)

        # Return most likely prediction and confidence scores
        response = {
            'rf_result': {
                'most_likely_class': rf_most_common[0],
                'confidence_score': round(rf_confidence, 2)
            },
            'svm_result': {
                'most_likely_class': svm_most_common[0],
                'confidence_score': round(svm_confidence, 2)
            }
        }
        return jsonify(response)

    except json.JSONDecodeError as e:
        return jsonify({'error': f'JSON parsing error: {str(e)}'}), 400
    except KeyError as e:
        return jsonify({'error': f'Missing key in data: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
