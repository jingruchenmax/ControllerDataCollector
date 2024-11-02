using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class VRControllerDataCollector : MonoBehaviour
{
    // Struct to hold VR data
    private struct VRData
    {
        public float timestamp;
        public Vector3 hmdPosition;
        public Quaternion hmdRotation;
        public Vector3 leftControllerPosition;
        public Quaternion leftControllerRotation;
        public Vector3 rightControllerPosition;
        public Quaternion rightControllerRotation;

        public VRData(float time, Vector3 hmdPos, Quaternion hmdRot, Vector3 leftPos, Quaternion leftRot, Vector3 rightPos, Quaternion rightRot)
        {
            timestamp = time;
            hmdPosition = hmdPos;
            hmdRotation = hmdRot;
            leftControllerPosition = leftPos;
            leftControllerRotation = leftRot;
            rightControllerPosition = rightPos;
            rightControllerRotation = rightRot;
        }

        public override string ToString()
        {
            return $"{timestamp}," +
                   $"{hmdPosition.x},{hmdPosition.y},{hmdPosition.z}," +
                   $"{hmdRotation.w},{hmdRotation.x},{hmdRotation.y},{hmdRotation.z}," +
                   $"{leftControllerPosition.x},{leftControllerPosition.y},{leftControllerPosition.z}," +
                   $"{leftControllerRotation.w},{leftControllerRotation.x},{leftControllerRotation.y},{leftControllerRotation.z}," +
                   $"{rightControllerPosition.x},{rightControllerPosition.y},{rightControllerPosition.z}," +
                   $"{rightControllerRotation.w},{rightControllerRotation.x},{rightControllerRotation.y},{rightControllerRotation.z}";
        }
    }

    public Transform hmdTransform; // Transform for HMD
    public Transform leftControllerTransform; // Transform for left controller
    public Transform rightControllerTransform; // Transform for right controller

    private List<VRData> collectedData;
    private bool isRecording;
    private float startTime;

    // Public methods to start and stop data collection
    public void StartRecording()
    {
        collectedData = new List<VRData>();
        startTime = Time.time;
        isRecording = true;
        Debug.Log("Data recording started.");
    }

    public void StopRecording()
    {
        isRecording = false;
        SaveDataToFile();
        Debug.Log("Data recording stopped.");
    }

    private void FixedUpdate()
    {
        if (isRecording)
        {
            float currentTime = Time.time - startTime;

            Vector3 hmdPos = hmdTransform != null ? hmdTransform.position : Vector3.zero;
            Quaternion hmdRot = hmdTransform != null ? hmdTransform.rotation : Quaternion.identity;

            Vector3 leftPos = leftControllerTransform != null ? leftControllerTransform.position : Vector3.zero;
            Quaternion leftRot = leftControllerTransform != null ? leftControllerTransform.rotation : Quaternion.identity;

            Vector3 rightPos = rightControllerTransform != null ? rightControllerTransform.position : Vector3.zero;
            Quaternion rightRot = rightControllerTransform != null ? rightControllerTransform.rotation : Quaternion.identity;

            VRData data = new VRData(currentTime, hmdPos, hmdRot, leftPos, leftRot, rightPos, rightRot);
            collectedData.Add(data);
        }
    }

    private void SaveDataToFile()
    {
        if (collectedData == null || collectedData.Count == 0)
        {
            Debug.LogWarning("No data to save.");
            return;
        }

        string fileName = $"{System.DateTime.Now:yyyy-MM-dd_HH-mm-ss}.csv";
        string filePath = Path.Combine(Application.persistentDataPath, fileName);

        using (StreamWriter writer = new StreamWriter(filePath))
        {
            // Write header
            writer.WriteLine("timestamp," +
                             "hmd_pos_x,hmd_pos_y,hmd_pos_z," +
                             "hmd_rot_w,hmd_rot_x,hmd_rot_y,hmd_rot_z," +
                             "left_controller_pos_x,left_controller_pos_y,left_controller_pos_z," +
                             "left_controller_rot_w,left_controller_rot_x,left_controller_rot_y,left_controller_rot_z," +
                             "right_controller_pos_x,right_controller_pos_y,right_controller_pos_z," +
                             "right_controller_rot_w,right_controller_rot_x,right_controller_rot_y,right_controller_rot_z");

            // Write data
            foreach (var data in collectedData)
            {
                writer.WriteLine(data.ToString());
            }
        }

        Debug.Log($"Data saved to {filePath}");
    }
}
