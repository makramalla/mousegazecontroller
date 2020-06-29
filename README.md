# Computer Pointer Controller

This Project uses a series of OpenVino models to detect a person's eye's gazes either form a video or from a WebCam to control PC's mouse movements.
The models used to achieve this setup are the following:
- [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
- [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
- [Landmarks Detection model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
- [Gaze Estimation Model](https://docs.openvinotoolkit.org/2019_R1/_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

The Algorithm Pipline looks like this:
![Pipeline](https://github.com/makramalla/mousegazecontroller/blob/master/Pipeline.png?raw=true)

## Project Set Up and Installation
This Project was crated on an Ubuntu 18.04 Desktop VM, but can also be used on Windows, while making sure the correct software and packages are properly installed.
Initially, OpenVion has to be installe on the created Linux Envioment whihc can be found in [this link](https://docs.openvinotoolkit.org/2018_R5/_docs_install_guides_installing_openvino_linux.html)

After cloning this repository the user would have to simply run the prepare.sh file which makes sure that all the required packages are installed and the models are properly downloded.:
```bash
./prepare.sh
```
Note: if certain packages are not installed the script may ask for sudo authentication.

Once everything is in place the user can go ahead and start running the app.py script which has the following required options:
- input_type: can either be video or CAM
-- in case the input type is video an input_file has to be provided
- mode_precision: FP16 or FP32
- models_folder: the path where all the models are downloaded (in the scenario where the prepare.sh was used the path would be <path_to_repo>/models/intel/)
- device: the hardware used for model loading and inference (CPU, GPU,.. etc)

## Demo
As am example for running the project:
```bash
python3 app.py --input_type video --input_file bin/demo.mp4  --model_precision FP16 --models_folder models/intel/ --device CPU
```
This starts the projects and provides two different outputs:
- Controlling the mouse movements using the Webcam **OR** produces an output video that displays the coordinates of the mouse based on the processed input video
![GazeEstimationCoordinates](https://github.com/makramalla/mousegazecontroller/blob/master/Gaze-Coordinates.png?raw=true)
- Model Loading and Inference Time for eeach model for further analysis
```bash
The model load time for the FaceDetection Model is 0.217 seconds
The model load time for the LandMarks Model is 0.093 seconds
The model load time for the HeadPoseEstimation Model is 0.11 seconds
The model load time for the GazeEstimation Model is 0.132 seconds
The model inference time for the FaceDetection Model is 0.027 seconds
The model inference time for the LandMarks Model is 0.001 seconds
The model inference time for the HeadPose Model is 0.003 seconds
The model inference time for the GazeEstimation Model is 0.002 seconds
```

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
