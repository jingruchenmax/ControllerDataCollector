using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using System.Text;
using Newtonsoft.Json;
using TMPro;

public class RightCTestDataSender : MonoBehaviour
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
    public TextMeshPro text;

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
        StartCoroutine(SendDataToServer());
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

    private IEnumerator SendDataToServer()
    {
        if (collectedData == null || collectedData.Count == 0)
        {
            Debug.LogWarning("No data to send.");
            yield break;
        }

        // Create a JSON payload using Newtonsoft.Json
        List<Dictionary<string, float>> formattedData = new List<Dictionary<string, float>>();
        foreach (var entry in collectedData)
        {
            Dictionary<string, float> dataPoint = new Dictionary<string, float>
            {
                { "right_controller_pos_x", entry.rightControllerPosition.x },
                { "right_controller_pos_y", entry.rightControllerPosition.y },
                { "right_controller_pos_z", entry.rightControllerPosition.z },
                { "right_controller_rot_w", entry.rightControllerRotation.w },
                { "right_controller_rot_x", entry.rightControllerRotation.x },
                { "right_controller_rot_y", entry.rightControllerRotation.y },
                { "right_controller_rot_z", entry.rightControllerRotation.z }
            };
            formattedData.Add(dataPoint);
        }

        string jsonData = JsonConvert.SerializeObject(new { data = formattedData });
        Debug.Log(jsonData);

        using (UnityWebRequest www = new UnityWebRequest("http://127.0.0.1:5000/classify", "POST"))
        {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonData);
            www.uploadHandler = new UploadHandlerRaw(bodyRaw);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");

            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                Debug.Log("Server Response: " + www.downloadHandler.text);
                text.text = www.downloadHandler.text;
            }
            else
            {
                Debug.LogError("Error: " + www.error);
                text.text = www.error;
            }
        }
    }
}
