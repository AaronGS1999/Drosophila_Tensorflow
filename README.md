### DROSOPHILA & TENSORFLOW

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/Imagen_google_colab.png">
</p>

Compilation of steps to follow and tools to generate models to detect Drosophila melanogaster mutants. 
---
### 1.) DATASET PREPARATION

#### 1.1) Photo taking:

To take the photos we use a digital microscope Bysameyee 8-SA-00  to get photos with more details, a Canon EOS 70D camera with a 100 mm macro objective f/2.8L lens because it will be the camera used to perform the experiment and cardboard of different colours also used for the backgrounds (improves accuracy). 

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/photo_taking.jpg" | width=600 >
</p>
The images that were taken were divided into three different folders, one for wild flies, others for white flies and another for photos with both types of flies.

Note: The images obtained with the canon camera, which are considerably larger than those of the digital microscope, gave problems during the training, so it was decided to divide the photograph into 20 pieces with the python image slicer library. This number of fragments was to obtain images of a size similar to that obtained with the digital microscope

#### 1.2) Preparation of the folders:

using a python script (https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/14ea96aab87be04cf95a98b9fe272334f8ef5218/DATA%20PREPARATION/split_folder.py), the images from each folder (wild, white and both flies) were divided into three other folders (Training, validation and test) using the same proportions: 80% for training, 10% for validation and 10 % For tests. This was done mainly to avoid biases such as there being no hard-to-identify photos in the validation or a balanced representation of the types of flies in each group of images. 

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/split_folder.jpg" | width=600 >
</p>

#### 1.3) Labeling:

The final step in preparing the dataset is to indicate what the AI needs to learn. To perform this task, the opensource program LabelImage was used (https://github.com/tzutalin/labelImg).  With this program, the flies to be identified and the type (wild type or white type) are labeled in a graphical interface and then an .xml file can be saved where the label and the coordinates of the flies in the image are recorded in pascal voc format. This must be done for the images contained in the training and validation folder.

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/labeling.jpg" | width=600 >
</p>

#### 1.4) Increased training efficiency (optional but strongly recommended):

In order to increase the training performance, two techniques were applied to the dataset. Image fragments were collaged automatically  (Improves detection of small targets) and noise was artificially introduced into photos (Improves resilience to photo artifacts) using the rooboflow web platform (https://roboflow.com). After applying these improvements, in our case, we got 1251 photos for training and 55 photos for validation.

---

### 2.) TRAINING

#### 1.1) Google colab option (easier to use):

To use this option, you only need a gmail account and upload files to drive with a specific structure. Within a folder you must also include another called "train" where the photos intended for training will be with their respective .xml files and another called "val" where the photos for validation and their respective .xml files will be included.

The next step is to open the link to the google colab with the code for the training: https://colab.research.google.com/drive/1SdoGigd8u9fq0PAgp6AqXftFHUAuaNqb?usp=sharing

You need to change the runtime settings. Go in the top menu to runtime > change runtime type and enable GPU hardware acceleration. It should look like this:



---
