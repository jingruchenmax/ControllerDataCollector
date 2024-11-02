import pandas as pd
import json

# Sample JSON data (replace with the actual JSON data you are working with)
json_data = '''
[
    {
        "timestamp": 0.018966,
        "right_controller_pos_x": 0.157347,
        "right_controller_pos_y": 1.091343,
        "right_controller_pos_z": 0.218500,
        "right_controller_rot_w": 0.933954,
        "right_controller_rot_x": -0.340684,
        "right_controller_rot_y": 0.097294,
        "right_controller_rot_z": -0.046893
    },
    {
        "timestamp": 0.038966,
        "right_controller_pos_x": 0.157504,
        "right_controller_pos_y": 1.091574,
        "right_controller_pos_z": 0.219008,
        "right_controller_rot_w": 0.935178,
        "right_controller_rot_x": -0.336572,
        "right_controller_rot_y": 0.098886,
        "right_controller_rot_z": -0.048821
    }
]
'''

# Parse JSON string to a Python object (list of dictionaries)
try:
    data = json.loads(json_data)
    
    # Check if data is in the correct format (list of dictionaries)
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        # Convert to DataFrame
        df = pd.DataFrame(data)
        print("Converted DataFrame:")
        print(df)
    else:
        print("Error: JSON data is not in the expected format (list of dictionaries).")

except json.JSONDecodeError as e:
    print("Error decoding JSON:", e)
