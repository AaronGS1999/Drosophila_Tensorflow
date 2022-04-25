### DROSOPHILA & TENSORFLOW

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/Imagen_google_colab.png">
</p>

Compilation of steps to follow and tools to generate models to detect Drosophila melanogaster mutants. 
---
### 1.) DATASET PREPARATION

#### 1.1) Photo taking

To take the photos we use a digital microscope Bysameyee 8-SA-00  to get photos with more details, a Canon EOS 70D camera with a 100 mm macro objective f/2.8L lens because it will be the camera used to perform the experiment and cardboard of different colours also used for the backgrounds (improves accuracy). 

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/photo_taking.jpg" | width=600 >
</p>
The images that were taken were divided into three different folders, one for wild flies, others for white flies and another for photos with both types of flies.

Note: The images obtained with the canon camera, which are considerably larger than those of the digital microscope, gave problems during the training, so it was decided to divide the photograph into 20 pieces with the python image slicer library. This number of fragments was to obtain images of a size similar to that obtained with the digital microscope

#### 1.2) Preparation of the folders

using a python script (), the images from each folder (wild, white and both flies) were divided into three other folders (Training, validation and test) using the same proportions: 80% for training, 10% for validation and 10 % For tests. This was done mainly to avoid biases such as there being no hard-to-identify photos in the validation or a balanced representation of the types of flies in each group of images. 

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/split_folder.jpg" | width=600 >
</p>

---

