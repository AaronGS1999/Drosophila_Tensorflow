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
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/photo_taking.jpg" | width=800 >
</p>

The images that were taken were divided into three different folders, one for wild flies, others for white flies and another for photos with both types of flies.

Note: The images obtained with the canon camera, which are considerably larger than those of the digital microscope, gave problems during the training, so it was decided to divide the photograph into 20 pieces with the python image slicer library (https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/4a60bbc513e815d96bb4fb950f312cb4625300fe/DATA%20PREPARATION/slicer.py). This number of fragments was to obtain images of a size similar to that obtained with the digital microscope

#### 1.2) Preparation of the folders:

using a python script (https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/14ea96aab87be04cf95a98b9fe272334f8ef5218/DATA%20PREPARATION/split_folder.py), the images from each folder (wild, white and both flies) were divided into three other folders (Training, validation and test) using the same proportions: 80% for training, 10% for validation and 10 % For tests. This was done mainly to avoid biases such as there being no hard-to-identify photos in the validation or a balanced representation of the types of flies in each group of images. 

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/split_folder.jpg" | width=800 >
</p>

#### 1.3) Labeling:

The final step in preparing the dataset is to indicate what the AI needs to learn. To perform this task, the opensource program LabelImage was used (https://github.com/tzutalin/labelImg).  With this program, the flies to be identified and the type (wild type or white type) are labeled in a graphical interface and then an .xml file can be saved where the label and the coordinates of the flies in the image are recorded in pascal voc format. This must be done for the images contained in the training and validation folder.

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/labeling.jpg" | width=800 >
</p>

#### 1.4) Increased training efficiency (optional but strongly recommended):

In order to increase the training performance, two techniques were applied to the dataset. Image fragments were collaged automatically  (Improves detection of small targets) and noise was artificially introduced into photos (Improves resilience to photo artifacts) using the rooboflow web platform (https://roboflow.com). After applying these improvements, in our case, we got 1251 photos for training and 55 photos for validation.

---

### 2.) TRAINING

#### 2.1) Google colab option (easier to use):

To use this option, you only need a gmail account and upload files to google Drive with a specific structure. Within a folder you must also include another called "train" where the photos intended for training will be with their respective .xml files and another called "val" where the photos for validation and their respective .xml files will be included.

The next step is to open the link to the google colab with the code for the training: https://colab.research.google.com/drive/1SdoGigd8u9fq0PAgp6AqXftFHUAuaNqb?usp=sharing

You need to change the runtime settings. Go in the top menu to runtime > change runtime type and enable GPU hardware acceleration. It should look like this (Don't forget to save the changes):

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/runtime_setting.png">
</p>

Once this option is adjusted, you will be able to connect to the environment and after this you must mount your Drive in the environment by pressing the icon shown in the following image:

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/drive_setting.png">
</p>

Then, you must check that the content of the variable "training_path" and that of "validation_path" corresponds to the paths of your files. With the variable "tflite_filename" you can set the name of the .tflite file that will be generated after training:

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/path_setting.png">
</p>

Now you will have to choose the model you want to train. You can do this in the following part of the code:

      spec = model_spec.get('efficientdet_lite2')
      
In our case we have used the efficientdet_lite2 model because it meets the necessary precision requirements and in an acceptable time. You can change the model by changing the number "2" to 0, 1, 2, 3, and 4. The lower the number, the faster the model will make inferences, but will generally achieve lower accuracy values. You should assess which model is best for you to use based on your dataset. Also slower but more accurate models will have a bigger impact on the resources that Google colab allows you to use.

After this, it only remains to adjust two variables, the number of epochs and the batch_size.

 - batch_size: Number of photos it takes at a time to train the network each time. If the number is 4, the photos will be passed 4 by 4 until all the photos of the dataset are passed
     
 - epochs: Number of times the entire dataset passes through the AI during training
 
 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/training.jpg" | width=800 >
</p>
     
You can do this setting in the following line of code: 

     model = object_detector.create(training_data, model_spec=spec, epochs=120, batch_size=16, train_whole_model=True, validation_data=validation_data)
     
In our case, the best results are achieved with a number of epochs = 120 and a batch_size = 16. Note that a very high number of epochs can cause overfitting and make the model not generalize well and that a higher number of batch_sizes can improve model accuracy but will have a larger impact on google colab resources and may exceed the GPU memory limit.

At this time you are ready to run the cells sequentially using the play icon to the left of each cell. As a result you will get a .tflite file where the trained model is saved and you and you can download it to your computer.

Note: you cannot skip any cells. If this happens, the code will not work and you will have to restart the environment and execute each cell sequentially again.

After training, the model is evaluated before and after it is exported to a .tflite model. You will see these results directly in the google colab notebook and you should pay attention that the values do not differ too much. If this happens, something strange has happened in the training or export process and you will need to adjust your training and validation files and repeat the process.

Note: The mean average precision (mAP) was calculated using COCO (https://cocodataset.org/)

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/validation.jpg" | width=800 >
</p>

#### 2.2) Local training option (very dependent on the hardware you have):

If you have a good processor and above all, if you have an NVIDIA graphics card with enough memory for your interests, you could use this method as an alternative.

In order to use the GPU for training, it is necessary to install specific drivers for the graphics card and the appropriate python tensorflow package (Take a look here: https://www.tensorflow.org/install/source). In our case we use a laptop with windows operating system and an NVIDIA GeForce 1660 Ti laptop GPU and we install CUDA 11.2, cuDNN 8.1 and Bazel 3.7.2 to work with Tensorflow 2.6.0 package. 

With the prerequisites installed, the next step is to install the necessary packages with the appropriate versions. To work more comfortably we use Anaconda Navigator (https://www.anaconda.com/products/distribution) but any virtual environment manager can be used.

Python version and necessary packages versions:
 - Python 3.8.12
 - tflite-model-maker 0.3.4
 - tflite-support 0.3.1
 - tensorflow 2.6.0
 - keras 2.6.0
 - tensorflow-estimator 2.6.0

Once the virtual environment has been adjusted, you can execute the training script (https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/57746fd49055ac4c75a1ff1f284ce067cdd9da22/TRAINING/load_training_validation.py) to carry out the training. The code is practically the same as the one explained in the Google Colab version, except that in this case the paths are local directories on the computer (update script paths).

---

### 3.) IMAGE PROCESSING

#### 3.1) Google colab option (easier to use):

Link to the google colab: https://colab.research.google.com/drive/1yYFuL3nnxHVXfjSvsRz-pFrDTsFPO2Y7?usp=sharing

To use this option, you will again need a gmail account and a specific folder structure. it is necessary that you have a folder called "Image_Processing_Approach" in your Google Drive with the image you want to process (default: "01.JPG") and the trained model (default: drosophila_lite2_epochs120_batch16_img1251_wild_white_v7_V2.tflite). Finally you should also have a folder inside Image_Processing_Approach, called: "processed". The "processed" folder must be empty and google drive does not allow empty folders to be uploaded, so a code has been included that creates this folder in the Google Colab notebook.

---
