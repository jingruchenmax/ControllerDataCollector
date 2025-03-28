using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using System.Text;
using Newtonsoft.Json;
using TMPro;
using Newtonsoft.Json.Linq;

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
                { "left_controller_pos_x", entry.leftControllerPosition.x },
                { "left_controller_pos_y", entry.leftControllerPosition.y },
                { "left_controller_pos_z", entry.leftControllerPosition.z },
                { "left_controller_rot_w", entry.leftControllerRotation.w },
                { "left_controller_rot_x", entry.leftControllerRotation.x },
                { "left_controller_rot_y", entry.leftControllerRotation.y },
                { "left_controller_rot_z", entry.leftControllerRotation.z },
                { "hmd_pos_x", entry.hmdPosition.x },
                { "hmd_pos_y", entry.hmdPosition.y },
                { "hmd_pos_z", entry.hmdPosition.z },
                { "hmd_rot_w", entry.hmdRotation.w },
                { "hmd_rot_x", entry.hmdRotation.x },
                { "hmd_rot_y", entry.hmdRotation.y },
                { "hmd_rot_z", entry.hmdRotation.z },
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

            // Assuming text is a TextMeshPro object
            string responseText = www.downloadHandler.text;

            // Parse the JSON response to make it more readable
            try
            {
                JObject jsonResponse = JObject.Parse(responseText);
                string formattedText = "";

                foreach (var result in jsonResponse)
                {
                    formattedText += $"{result.Key}:\n";
                    JObject resultDetails = (JObject)result.Value;

                    foreach (var detail in resultDetails)
                    {
                        formattedText += $"- {detail.Key}: {detail.Value}\n";
                    }

                    formattedText += "\n"; // Add spacing between sections
                }

                // Assign the formatted text to the TextMeshPro object
                text.text = formattedText;
            }
            catch (System.Exception e)
            {
                // Handle any parsing errors and display the raw text if parsing fails
                text.text = "Error parsing response:\n" + e.Message + "\nRaw response:\n" + responseText;
            }
        }
    }
}
