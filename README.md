Code for training and testing are in the notebooks stage1.ipynb and stage2.ipynb. The final step to generate the csv files can be found in final.ipynb. Stage1.ipynb and stage2.ipynb generate a lot of intermediate data files, and doing so will take lots of time. Some of these data files were too large for GitHub, so they can be downloaded with this link : https://drive.google.com/open?id=1dhmdiUBOSm7D9tQvZUbXFmgkGTNRRCgs

When extracted, you should have the directory structure as follows:
`
    .
    
    ├── Final.ipynb
    ├── README.md
    ├── faster_rcnn_resnet101_kitti_2018_01_28
    │   ├── checkpoint
    │   ├── frozen_inference_graph.pb
    │   ├── model.ckpt.data-00000-of-00001
    │   ├── model.ckpt.index
    │   ├── model.ckpt.meta
    │   ├── pipeline.config
    │   └── saved_model
    │       ├── saved_model.pb
    │       └── variables
    ├── model_lib.py
    ├── stage1
    │   ├── calculated_bbox-test.pkl
    │   ├── checkpoint
    │   ├── graph.pbtxt
    │   ├── inference_graph
    │   │   ├── checkpoint
    │   │   ├── frozen_inference_graph.pb
    │   │   ├── model.ckpt.data-00000-of-00001
    │   │   ├── model.ckpt.index
    │   │   ├── model.ckpt.meta
    │   │   ├── pipeline.config
    │   │   └── saved_model
    │   │       ├── saved_model.pb
    │   │       └── variables
    │   ├── label-map.pbtxt
    │   ├── model.ckpt-95036.data-00000-of-00001
    │   ├── model.ckpt-95036.index
    │   ├── model.ckpt-95036.meta
    │   ├── pipeline.config
    │   ├── test.record
    │   └── train.record
    ├── stage1.ipynb
    ├── stage2
    │   ├── X_test.npy
    │   ├── X_test_eval.npy
    │   ├── X_train.npy
    │   ├── Y_predict.txt
    │   ├── Y_predict_eval.txt
    │   ├── Y_test.npy
    │   ├── Y_train.npy
    │   ├── Y_truth.txt
    │   ├── calculated_bbox-test.pkl
    │   ├── filename_map.npy
    │   ├── filename_map_eval.npy
    │   ├── model.h5
    │   ├── model_architecture.json
    │   ├── results-trained-resnet101-task1.csv
    │   ├── results-trained-resnet101-task2.csv
    │   └── template.csv
    └── stage2.ipynb
`
# Overview

For this project, we decided to split the classification problem into two stages. In stage 1, we used an existing algorithm to predict bounding boxes and labels for vehicles in each image frame. Next, in stage 2, we used the predicted bounding boxes detected in stage 1, point cloud statistics, and predicted labels to predict the distance and angle to the camera. With the distance and angle for all detected vehicles in each image from stage 2 and the labels from stage 1, we can solve task 1 and task 2 of the project. Both stages used TensorFlow 1.15 on Google Colab. 

## Stage 1

Stage 1 generates a list of 2D bounding boxes and their respective labels from an input image. To do this, we used the TensorFlow Object Detection API with the Faster-RCNN-ResNet101 found in the TensorFlow Object Detection Model Library. This network was pretrained on the Kitti dataset to aid in faster training for this project. 

To train the network for our purposes, we split the images in the trainval folder 80/20 to test and validate the Faster-RCNN-ResNet101 network on the given data. The ground truth bounding boxes in these images were determined from the bbox.bin file for each image. Training and evaluation data were stored as .record files for the TensorFlow Object Detection API to recognize. 

Training was done on Google Colab until the validation accuracy leveled off at around 10000 steps.


## Stage 2

Before running stage 2, we determined point cloud points that lay within each bounding box. From this, stage 2 inputs the bounding box coordinates (xmin, xmax, ymin, ymax), point cloud statistics for the points that lay within the bounding box (mean, median, min, max, standard deviation), and label (0-3) for a total of 10 scalar inputs. This stage outputs the distance and angle for a bounding box. 

We created a deep neural network and trained/evaluated it using the data in the trainval folder, split 80/20. This was done independently withstage 1. 

# Running the Notebooks

Only final.ipynb needs to be run to generate the .csv files.

## Final.ipynb

All the files needed for this notebook are existing in GitHub.  

First, mount a Google Drive if needed. 

Next, set the file path constants PROJECT_ROOT and TEST_FILES to the full paths of the project root and the test files (i.e. data-2019/test) respectively. 

Next, run imports and evaluate sections to generate the csv files. 

## Stage1.ipynb

The first cell loads the Google Drive where the project files are located. Run the Install Models section to install the Object Detection API and set the file path constants to where the project files and data are located. 

Run the cells in the “Create Record Files for Training” section to generate the .record files need for the Object Detection API. This takes a while on Colab so existing files are stored in the .zip file.

Run the cells in the “Train Model” section to train the model. This takes a very long time, so we have stored the frozen model we used in the .zip file. 

Run the cells in the “Save Model and Run on Test Images” section to save the model and run the model on the test images (test folder). The results from this were stored in “calculated_bbox_train.pkl” and used in stage 2. 

Run the “Visualize Single Image” section to visualize all the bounding boxes detected on a random image found in the test folder. 

## Stage2.ipynb

The first cell loads the Google Drive where the project files are located. Be sure to set the file path constants to where the project files and data are located. 

The training section first generates training for stage 2 and then trains the model. 

The evaluation section first preprocesses the output data from stage 1 and then generate the CSV files for submission. 
