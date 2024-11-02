# ControllerDataController

This project is designed for collecting HMD and controller data in Unity (position, rotation) for sktime ML testing.

## System Requirements

- Unity version: 2022.3.8f1
- Required Package: Meta All-in-One
- Tested with: Meta Quest 3

## Unity Setup

Connect to Meta using Meta Quest Link.

1. **Waiting Room**
   Ensures that the start scene trigger is recorded.

2. **Gesture Collection Scene**
   Press the button to start and stop recording (A for the right controller, X for the left controller). During the trigger period, all data will be recorded in the provided example CSV file. Once recording is complete, the data will be saved to the CSV to prevent data loss in case of a Unity failure.

3. **Special Effect Scenes**
   Visual, audio, and haptic feedback are associated with tracing and classification results.

4. **Generative AI Scene**
   Classification results trigger the generative AI system.

## Preliminary Pattern Naming
![](image.png)

