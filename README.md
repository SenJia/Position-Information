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

# Update 07/08/2021, 
1. Fixed some typos 
2. The dataloader has been rewritten, now only the data folder is needed to load all images, e.g., $abs_path_folder/a.jpg, pass the absolute path of the folder. Similarly, pass the folder of the test image as the second argument.
```
python train_network.py folder $abs_train_folder $abs_test_filder
```
3. The evaluation code is added to the ``train_network.py'', the MSE loss and the Spearman score will be shown after each evaluation process.


In our work, we train the whole system on the DUT-S dataset and validate on the PASCAL-S dataset, they both are originally used for salient object detection. The position information we explored is content-agnostic, so any natural images can be used. You might want to avoid the ImageNet dataset, because the backbone(vgg) is commonly pre-trained on the data.
