# *What Objects Are Where?* Object Detection With Rich Feature Hierarchies

Object detection is one of the most fundamental and challenging problems in computer vision. It deals with detecting instances of a certain class in digital images and forms the basis of many other computer vision tasks such as instance segmentation, image captioning, object tracking, etc. 

## Traditional Object Detectors

Most of the early object detection algorithms were built based on handcrafted features. Due to the lack of effective image representation at that time, it was necessary to design sophisticated feature representations and a variety of speed up skills to exhaust the usage of limited computing resources.

## CNN based Two-stage Detector: R-CNN

Object detection had reached a plateau after 2010 as the performance of hand-crafted features became saturated. Small gains were obtained by building ensemble systems and minor variants of successful models.

In 2012, the world saw the rebirth of convolutional neural networks. A deep convolutional network is able to learn robust and high-level feature representations of an image. In 2014, R.Girshick *et. al.* proposed the Regions with CNN features (RCNN) for object detection and since then, object detection started to evolve at an unprecedented speed.

## Region-based Convolutional Neural Network

Region-based ConvNet is a natural combination of heuristic region proposal method and ConvNet feature extractor. From an image, around 2000 bounding box proposals are generated using selective search. Those proposed regions are cropped and warped to a fixed-size 227 x 227 image. AlexNet is then used to extract a feature vector for each warped image. An SVM model is then trained to classify the object in the warped image using its features.

<img src="assets/rcnn_overview.jpg" alt="Logo">

R-CNNs are composed of three main parts.

### A. Selective Search

Selective search is performed on the input image to select multiple high-quality proposed regions. These proposed regions are generally selected on multiple scales and have different shapes and sizes. The category and ground-truth bounding box of each proposed region is labelled.

### B. Feature Extraction

A pre-trained CNN is selected and placed, in truncated form, before the output layer. It transforms each proposed region into the input dimensions required by the network and uses forward computation to output the fetures extracted from the proposed regions.

### C. Object Classification

The features and labelled category  of eeach proposed region are combined as an example to train multiple support vector machines for object classification. In our case, each support vector machine is used to determine whether an example belongs to a certain category or not.

## To Run The Demo

#### 01. Clone the Repository

```bash
git clone https://github.com/Computer-Vision-IIITH-2021/project-revision.git
```

#### 02. Setting Up Virtual Environment
```bash
conda create --name envname python=3.8
conda activate envname
```
Ensure that you install all the dependencies in the virtual environment before running the program. We have used `Python 3.8` during the development process. Do ensure that you have the same version before running the code.

#### 03. Running Inference On Local Machine
```bash
cd src
python3 test.py
```
Do note that the training process may take several hours. The team members used the Ada High Performance Cluster of IIIT Hyderabad for training the model.

## Team Members

<table>
  <tr>
    <td align="center"><a href="https://github.com/<a href="https://github.com/doltonfernandes/"><img src="https://avatars1.githubusercontent.com/u/42113482?s=460&u=34e4c282db236d7adfbc6ff0176992cb973b426a&v=4" width="100px;" alt=""/><br /><sub><b>Dolton Fernandes</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/georg3tom"><img src="https://avatars0.githubusercontent.com/u/22193688?s=460&u=b4874125263dd8d3ba21a87e3f4f76d0fd0a825d&v=4" width="100px;" alt=""/><br /><sub><b>George Tom</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/narenakash"><img src="https://avatars3.githubusercontent.com/u/43748290?s=460&u=26bc998e91194730ff8d885ec34ee64690ec46c2&v=4" width="100px;" alt=""/><br /><sub><b>Naren Akash R J</b></sub></a><br /></td>
  <tr>
</table>

All the team members are undergraduate research students at the [Center for Visual Information Technology, IIIT Hyderabad](http://cvit.iiit.ac.in/), India.

## Licence and Citation
The software can only be used for personal/research/non-commercial purposes. To cite the original paper:
```
@INPROCEEDINGS{7410526,
  author={Girshick, Ross},
  booktitle={2015 IEEE International Conference on Computer Vision (ICCV)}, 
  title={Fast R-CNN}, 
  year={2015},
  volume={},
  number={},
  pages={1440-1448},
  doi={10.1109/ICCV.2015.169}}
```


