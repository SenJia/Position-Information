# Position-Information
This repo contains the code for our ICLR project "How Much Position Information Do Convolutional Neural Networks Encode?"
If you think this project is helpful to research please cite our paper:  
@inproceedings{Islam20,  
title={How much Position Information Do Convolutional Neural Networks Encode?},  
author={Md Amirul Islam* and Sen Jia* and Neil D. B. Bruce},  
booktitle={International Conference on Learning Representations},  
year={2020},  
url={https://openreview.net/forum?id=rJeB36NKvB}  
}  

The "train_network.py" and "evaluation.py" are the files for training and testing. All the settings are commonly used in other Pytorch vision projects, but our code load the pre-trained backbone(VGG or ResNet) as a default behavior. Because our study attemts to demystify if a pre-trained model could contain the position information, the weight of the backbone is freezed. The simple readout is trainable in order to extract position information from the backbone as much as it can. All the model definitions are under the folder "models". The synthetic images(black, white, noise) and the groundtruth(horizontal, vertical) are under the folder "synthetic".

| Horizontal | Vertical | Gaussian | Hor Stripes | Ver Stripes |
| ------------- | ------------- | ----------| ----------| -------- |
| <img src="https://github.com/SenJia/Position-Information/raw/master/synthetic/groundtruth/gt_hor.png" width="100px" height="100px"> |  <img src="https://github.com/SenJia/Position-Information/raw/master/synthetic/groundtruth/gt_ver.png" width="100px" height="100px">| <img src="https://github.com/SenJia/Position-Information/raw/master/synthetic/groundtruth/gt_gau.png" width="100px" height="100px">| <img src="https://github.com/SenJia/Position-Information/raw/master/synthetic/groundtruth/gt_horstp.png" width="100px" height="100px"> | <img src="https://github.com/SenJia/Position-Information/raw/master/synthetic/groundtruth/gt_verstp.png" width="100px" height="100px"> |

# Training
To train the model(decoder, readout), you need to specify your training and validation data in the "train_network.py". Our code follows a Caffe-like style, a image root should be assigned to the variable "DATA_ROOT", a text file should be assigned to "train_file"(for the training code), or "val_file". Each line of the text file should contain the relative path of your images, e.g., "a/b/image1.png". The relative path will be concatenated with the data folder, "DATA_DIR" / "a/b/image.png". You can select the types of the groundtruth, see line 226 in the training code, or other settings in the args section, e.g. architecture, vgg, resnet, img indicates the readout will be trained on the raw image directly.

# Evaluation
"evaluation.py" is used to load the pre-trained decoder (the backone is loaded of course) to predict on new images. Again, the variable "DATA_DIR" should be assigned to locate the root of your data, the text file is assigned in the variable "VAL_DATA"(uncomment it). By default, the code will load the synthetic data (white, black, noise) for comparison as well. An argument is necessary when running the code, the folder of your pre-trained decoder, e.g., "vgg_hor". The code will seek for the model file "decoder.pth.tar" automatically, and the type of groundtruth will be parsed from your folder name, "hor" in this case. Other arguments should be easy to understand based on the help information.

In our work, we train the whole system on the DUT-S dataset and validate on the PASCAL-S dataset, they both are originally used for salient object detection. The position information we explored is content-agnostic, so any natural images can be used. You might want to avoid the ImageNet dataset, because the backbone(vgg) is commonly pre-trained on the data.
