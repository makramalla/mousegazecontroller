"""
A lot of this code is inspired from the follwing Repos:

https://github.com/CJRiverFlow/final_project
https://github.com/Eslam26/Computer-Pointer-Controller/tree/master
https://github.com/lilliCao/mouse_controller

"""


import cv2
import time
import pandas as pd
import pyautogui

from src.input_feeder import InputFeeder
from src.face_detection import FaceDetection
from src.facial_landmarks_detection import LandmarkDetection
from src.head_pose_estimation import PoseEstimation
from src.gaze_estimation import GazeEstimation
from src.mouse_controller import MouseController
from argparse import ArgumentParser

video_path = "bin/demo.mp4"
models_path = "models/intel/"

#Dataframe for saving perfomance times
df = pd.DataFrame(0.0, columns=["face_detection","face_landmarks","headpose_estimation",
                  "gaze_estimation"],index=["loading_time","inference_time"])


def build_argparser():
    """
    Parse command line arguments.
    Return parser
    """
    parser = ArgumentParser()

    parser.add_argument("--input_type", required=True, type=str,
                        default="cam",
                        help="Type of video input, camera or file")
    parser.add_argument("--models_folder", required=True, type=str,
                        default="models/intel/",
                        help="Type of video input, camera or file")
    parser.add_argument("--input_file", required=False, type=str,
                        default="bin/demo.mp4",
                        help="Path to video file to be used")
    parser.add_argument("--device", required=True, type=str, default="CPU",
                        help="Specify the target device to infer on:CPU, GPU, FPGA or MYRIAD is acceptable")
    parser.add_argument("--model_precision", required=False, type=str,
                        default="FP32",
                        help="Model precision options: FP32, FP16")
    parser.add_argument("--mouse_precision", required=False, type=str,
                        default="high",
                        help="Set Mouse Precision")
    parser.add_argument("--mouse_speed", required=False, type=str,
                        default="fast",
                        help="Set Mouse Speed")
    return parser


def create_models(selected_precision,models_path, device):
    """
    Model options available are either FP16 or FP32
    """
    global fd, fl, hp, gz

    valid_values = {"INT8":"FP16-INT8",
                    "FP32":"FP32",
                    "FP16":"FP16"}

    #Applying user selection.
    if selected_precision in valid_values:
        fland_model = "landmarks-regression-retail-0009/{}/landmarks-regression-retail-0009"\
                    .format(valid_values[selected_precision])
        head_pose = "head-pose-estimation-adas-0001/{}/head-pose-estimation-adas-0001"\
                    .format(valid_values[selected_precision])
        gaze_model = "gaze-estimation-adas-0002/{}/gaze-estimation-adas-0002"\
                    .format(valid_values[selected_precision])
    else:
        raise ValueError('No valid model precision seleted')

    fd_model = "face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001"
    fd = FaceDetection(models_path+fd_model, device)
    fl = LandmarkDetection(models_path+fland_model, device)
    hp = PoseEstimation(models_path+head_pose, device)
    gz = GazeEstimation(models_path+gaze_model, device)


def get_load_time():
    global df

    #LOADING MODELS and BENCHMARKING
    #Face detection
    fd_start_load_time=time.time()
    fd.load_model()
    df["face_detection"]["loading_time"] = round(time.time()-fd_start_load_time,3)
    print("The model load time for the FaceDetection Model is %s seconds" % df["face_detection"]["loading_time"])

    #Facial landmarks detection
    fl_start_load_time=time.time()
    fl.load_model()
    df["face_landmarks"]["loading_time"] = round(time.time()-fl_start_load_time,3)
    print("The model load time for the LandMarks Model is %s seconds" % df["face_landmarks"]["loading_time"])

    #Head pose estimation
    hp_start_load_time=time.time()
    hp.load_model()
    df["headpose_estimation"]["loading_time"] = round(time.time()-hp_start_load_time,3)
    print("The model load time for the HeadPoseEstimation Model is %s seconds" % df["headpose_estimation"]["loading_time"])

    #Gaze estimation model
    gaze_start_load_time=time.time()
    gz.load_model()
    df["gaze_estimation"]["loading_time"] = round(time.time()-gaze_start_load_time,3)
    print("The model load time for the GazeEstimation Model is %s seconds" % df["gaze_estimation"]["loading_time"])




def inference_get_coordinates(batch):
    """
    This Funtion goes through each models and caluclates the inference time.

    It starts by detecting the face, and cropping it.
    Then cropping the eyes and accordingly their angle.
    Next is the gaze estimation time.
    Finally it calls another funciton to print the output on the video
    """
    global df

    output = batch.copy()


    #Inference for face detection
    fd_start_inference_time=time.time()
    face_coords, image = fd.predict(batch)
    df["face_detection"]["inference_time"] += round(time.time()-fd_start_inference_time,3)
    print("The model inference time for the FaceDetection Model is %s seconds" % df["face_detection"]["inference_time"])
    #Crop the detcted Face
    cropped_face_img = image[face_coords[0][1]:face_coords[0][3],
                        face_coords[0][0]:face_coords[0][2]]

    #GETTING CROPPED EYES and measure time from cropped face
    fl_start_inference_time=time.time()
    landmarks, left_eye, right_eye = fl.predict(cropped_face_img) #inference
    df["face_landmarks"]["inference_time"] += round(time.time()-fl_start_inference_time,3)
    print("The model inference time for the LandMarks Model is %s seconds" % df["face_landmarks"]["inference_time"])
    #GETTING POSE ANGLES and measure time from cropped face
    hp_start_inference_time=time.time()
    head_pose_angles = hp.predict(cropped_face_img) #inference
    df["headpose_estimation"]["inference_time"] += round(time.time()-hp_start_inference_time,3)
    print("The model inference time for the HeadPose Model is %s seconds" % df["headpose_estimation"]["inference_time"])
    #print(head_pose_output)

    #RUNNING GAZE ESTIMATION and measure time.
    gaze_start_inference_time=time.time()
    gaze_outputs = gz.predict(left_eye, right_eye, head_pose_angles) #inference
    df["gaze_estimation"]["inference_time"] += round(time.time()-gaze_start_inference_time,3)
    print("The model inference time for the GazeEstimation Model is %s seconds" % df["gaze_estimation"]["inference_time"])
    #Printing values on screen
    #print("gaze model outputs: ",gaze_outputs)
    output = cv2.resize(output, (1080, 600), interpolation = cv2.INTER_AREA)

    value_dic = {"Face Cropcoordinates: ": face_coords,
                    "eyex Coordinates: ":landmarks,
                    "Head pose angels: ":head_pose_angles,
                    "Gaze Estimation: ":gaze_outputs}

    #Prinitng on Screen
    y_pos = 10
    for value in value_dic:
        text = value+str(value_dic[value])
        y_pos+=20
        cv2.putText(output, str(text), (20,y_pos), cv2.FONT_HERSHEY_PLAIN, 0.5, (128, 0, 128), 1)

    cv2.imshow("Frame",output)
    cv2.waitKey(1)

    return gaze_outputs[0][0], gaze_outputs[0][1]


def main():
    args = build_argparser().parse_args()


    create_models(args.model_precision, args.models_folder, args.device)
    global df
    get_load_time()


    screen_res = pyautogui.size()
    pyautogui.moveTo(int(screen_res[0]/4), int(screen_res[1]/4), duration=1)

    mousecontroller = MouseController(precision=args.mouse_precision, speed=args.mouse_speed)

    feed=InputFeeder(input_type=args.input_type, input_file=args.input_file)
    feed.load_data()
    for batch in feed.next_batch():
        if batch is not None:
            #MOVING THE MOUSE
            x_coord, y_coord  = inference_get_coordinates(batch)
            mousecontroller.move(x_coord,y_coord)
        else:
            feed.close()
            break


    print("Done")

if __name__ == "__main__":
    main()
