#! /bin/bash

sudo apt-get install python-opencv
pip3 install pandas


MODELDIR="$(pwd)/models/"
DOWNLOADER="/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py"
if [ ! -d "$MODELDIR" ] 
then
    mkdir models
    echo "Created path: '$MODELDIR'"
fi

echo "Downloading models to: '$MODELDIR'"


python3 $DOWNLOADER --name face-detection-adas-binary-0001 --output_dir $MODELDIR
python3 $DOWNLOADER --name head-pose-estimation-adas-0001 --output_dir $MODELDIR
python3 $DOWNLOADER --name landmarks-regression-retail-0009 --output_dir $MODELDIR
python3 $DOWNLOADER --name gaze-estimation-adas-0002 --output_dir $MODELDIR


