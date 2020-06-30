import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

class LandmarkDetection:
    '''
    Class for the Facial Landmarks Detection Model.
    '''
    def __init__(self, model_name, device, extensions=None):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device= device
        self.request_id=0
        self.num_requests=1 
        self.ie = IECore()
        self.net = None
        self.exec_net = None
        self.frame_width = None
        self.frame_height = None
        self.extensions = extensions

        self.check_model()
        self.input_name=next(iter(self.net.inputs))
        self.input_shape=self.net.inputs[self.input_name].shape
        self.output_name=next(iter(self.net.outputs))
        self.output_shape=self.net.outputs[self.output_name].shape

    def check_model(self):
        try:
            self.net=self.ie.read_network(self.model_structure, self.model_weights)
        except:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

    def load_model(self):
        self.exec_net = self.ie.load_network(network=self.net, device_name=self.device, num_requests=self.num_requests)

    def predict(self, image):
        self.frame_width, self.frame_height = image.shape[1], image.shape[0]
        
        preprocessed_image = self.preprocess_input(image)
        input_dict={self.input_name: preprocessed_image}

        self.exec_net.requests[self.request_id].infer(input_dict)
        outputs = self.exec_net.requests[self.request_id].outputs[self.output_name]
        
        landmarks = self.preprocess_output(outputs)

        
        #Getting the left and right eyes using only two landmarks

        left_point = landmarks[1]
        right_point = landmarks[0]
        #Creating the eyses square using a "Radius" of 20
        radius = 20

        left_square = [(left_point[0]-radius, left_point[1]-radius),
                         (left_point[0]+radius, left_point[1]+radius)]
        right_square = [(right_point[0]-radius, right_point[1]-radius),
                         (right_point[0]+radius, right_point[1]+radius)]

        left_eye_crop = image[left_square[0][1]:left_square[1][1], 
                         left_square[0][0]:left_square[1][0]]

        right_eye_crop = image[right_square[0][1]:right_square[1][1], 
                          right_square[0][0]:right_square[1][0]]
        

        return landmarks, left_eye_crop, right_eye_crop
        

    def preprocess_input(self, image):
        input_img = cv2.resize(image, (self.input_shape[3], self.input_shape[2]), 
                                interpolation = cv2.INTER_AREA) 
        input_img = np.moveaxis(input_img, -1, 0)
        return input_img


    def preprocess_output(self, outputs):
        #Getting only eyes landmarks
        cords = [element[0][0] for element in outputs[0]]

        x0 = int(cords[0]*self.frame_width)
        y0 = int(cords[1]*self.frame_height)
        x1 = int(cords[2]*self.frame_width)
        y1 = int(cords[3]*self.frame_height)

        eyes_cords = [(x0,y0),(x1,y1)]    
        return eyes_cords