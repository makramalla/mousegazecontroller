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
- mode_precision: FP16,  FP32 or INT8
- models_folder: the path where all the models are downloaded (in the scenario where the prepare.sh was used the path would be <path_to_repo>/models/intel/)
- device: the hardware used for model loading and inference (CPU, GPU,.. etc)

## Demo
As am example for running the project:
```console
python3 app.py --input_type video --input_file bin/demo.mp4  --model_precision FP16 --models_folder models/intel/ --device CPU
```
This starts the projects and provides two different outputs:
- Controlling the mouse movements using the Webcam **OR** produces an output video that displays the coordinates of the mouse based on the processed input video
![GazeEstimationCoordinates](https://github.com/makramalla/mousegazecontroller/blob/master/Gaze-Coordinates.png?raw=true)
- Model Loading and Inference Time for eeach model for further analysis
```console
The model load time for the FaceDetection Model is 0.217 seconds
The model load time for the LandMarks Model is 0.093 seconds
The model load time for the HeadPoseEstimation Model is 0.110 seconds
The model load time for the GazeEstimation Model is 0.132 seconds
The model inference time for the FaceDetection Model is 0.027 seconds
The model inference time for the LandMarks Model is 0.001 seconds
The model inference time for the HeadPose Model is 0.003 seconds
The model inference time for the GazeEstimation Model is 0.002 seconds
```

## Benchmarks
The average model Loading Time after running the application several times is the following:
### Model Loading Time

#### FP32 in ms

| FaceDetection | LandMarks |HeadPoseEstimation  |GazeEstimation  |
| ------------- |:-------------:| -----:|-----:|
| 177   | 82 | 83| 113 |

#### FP16 in ms
| FaceDetection | LandMarks |HeadPoseEstimation  |GazeEstimation  |
| ------------- |:-------------:| -----:|-----:|
| 181    | 82 | 98| 125 |

#### INT8 in ms
| FaceDetection | LandMarks |HeadPoseEstimation  |GazeEstimation  |
| ------------- |:-------------:| -----:|-----:|
| 218     | 95 |161| 192|



### Model Inference Time FP32 in ms
#### FP32 in ms

| FaceDetection | LandMarks |HeadPoseEstimation  |GazeEstimation  |
| ------------- |:-------------:| -----:|-----:|
| 24    | 6 | 3| 2 |

#### FP16 in ms
| FaceDetection | LandMarks |HeadPoseEstimation  |GazeEstimation  |
| ------------- |:-------------:| -----:|-----:|
| 125     | 8 | 9| 14 |

#### INT8 in ms
| FaceDetection | LandMarks |HeadPoseEstimation  |GazeEstimation  |
| ------------- |:-------------:| -----:|-----:|
| 60     | 20 |12| 9|



In my case, The VM only had a CPU to test with. Further analysis can be made using differnt hardware
## Results
<MISSING>
### Edge Cases
This model was developed in a way to take only the first face it captures and uses is to get the eyes and landmarks
## Stand Out Suggestions
A suggestion fro improvement is to design it in a way to allow the program to take the main face in the captures video stream to be used. This can be done by cropping out the largest face with the most clear featrues.
As an addiotnal suggestion, a hand gesture may be introduced that would trigger mouse tracking and movemenets.



