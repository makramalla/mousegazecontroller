import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

class GazeEstimation:
    '''
    Class for the Gaze Estimation Model.
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
        self.extensions = extensions

        self.check_model()
        self.output_name=next(iter(self.net.outputs))
        

    def check_model(self):
        try:
            self.net=self.ie.read_network(self.model_structure, self.model_weights)
        except:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")


    def load_model(self):
        self.exec_net = self.ie.load_network(network=self.net, device_name=self.device, num_requests=self.num_requests)


    def predict(self, left_eye, right_eye, angles):
        input_dict={'head_pose_angles': angles,
                    'left_eye_image': self.preprocess_input(left_eye),
                    'right_eye_image':self.preprocess_input(right_eye)}    

        self.exec_net.requests[self.request_id].infer(input_dict)
        
        return self.exec_net.requests[self.request_id].outputs[self.output_name]


    def preprocess_input(self, image):
        input_img = cv2.resize(image, (60, 60), 
                                interpolation = cv2.INTER_AREA) 
        input_img = np.moveaxis(input_img, -1, 0)
        return input_img