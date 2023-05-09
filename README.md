### DROSOPHILA & TENSORFLOW

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/Imagen_google_colab.png">
</p>

How to generate an AI model trained to detect Drosophila melanogaster mutants 
---
### 1.) DATASET PREPARATION

#### 1.1) Photo taking:

To capture the pictures, we use a Bysameyee 8-SA-00 digital microscope which provides more detailed photos. Additionally, we use a Canon EOS 70D camera with a 100mm macro objective f/2.8L lens, as this camera will be used for the experiment. We also use cardboard of various colors as backgrounds, which helps improve accuracy.

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/photo_taking.jpg" | width=800 >
</p>

The captured images were sorted into three different folders: one for wild flies, one for white flies, and one for photos with both types of flies. However, there were some issues during the training phase with images taken using the Canon camera, which were significantly larger (5470x3072 pixels) than those taken with the digital microscope (1094x912 pixels). To address this issue, the python image slicer library (https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/4a60bbc513e815d96bb4fb950f312cb4625300fe/DATA%20PREPARATION/slicer.py) was used to divide the larger images into 20 pieces (1094x768 pixels) to obtain a similar size to that of the microscope images. To ensure consistency, photographs were taken under specific conditions, including the use of a simple grid of 20 cells (4 rows * 5 columns) designed in Word and printed on blue cardboard to provide better contrast for the fly colors. The camera was positioned at a 90º angle and a specific distance (to be added) from the grid, with the lens focused on the central cells.

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/camera_tripod.jpg" | width=300 >
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/scale.jpg" | width=600 >
</p>

#### 1.2) Preparation of the folders:

A python script (https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/14ea96aab87be04cf95a98b9fe272334f8ef5218/DATA%20PREPARATION/split_folder.py) was used to divide the images from each folder (wild, white, and both flies) into three separate folders (Training, Validation, and Test) using the same proportions: 80% for training, 10% for validation, and 10% for testing. This was done to ensure a balanced representation of each type of fly and to prevent biases such as the absence of hard-to-identify photos in the validation set. 

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/split_folder.jpg" | width=800 >
</p>

#### 1.3) Labeling:

The last step in dataset preparation is to define what the AI needs to learn. This was done using the open-source program LabelImage (https://github.com/tzutalin/labelImg). The program allows flies to be labeled and classified as wild type or white type in a graphical interface. An .xml file is generated in pascal voc format, which records the label and coordinates of the flies in the image. This labeling process is done for all images in the training and validation folders.

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/labeling.jpg" | width=800 >
</p>

#### 1.4) Increased training efficiency (optional but strongly recommended):

To enhance the training performance, two techniques were applied to the dataset using the roboflow web platform (https://roboflow.com). The first technique involved automatically combining image fragments to improve the detection of small targets. The second technique involved artificially introducing noise into the photos to enhance their resilience to photo artifacts. An example of this is the following image with 5% noise.

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/improvements_example.jpg" | width=800 >
</p>

After applying these improvements, in our case, we got 1251 photos for training and 55 photos for validation.

---

### 2.) TRAINING

#### 2.1) Training with Google colab:

To get started, all you need is a Gmail account and upload the files to Google Drive following a specific structure. Within a folder, you should create two more folders named "train" for the photos intended for training with their respective .xml files, and "val" for the photos and their respective .xml files intended for validation.

The next step is to open the Google Colab link (https://colab.research.google.com/drive/1SdoGigd8u9fq0PAgp6AqXftFHUAuaNqb?usp=sharing) that contains the code for the training. You need to change the runtime settings by going to the top menu and selecting runtime > change runtime type, and enabling GPU hardware acceleration. Don't forget to save the changes.

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/runtime_setting.png">
</p>

After adjusting the GPU hardware acceleration option, connect to the environment and mount your Drive by clicking on the icon shown in the image below:

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/drive_setting.png">
</p>

Next, you need to verify that the paths specified in the "training_path" and "validation_path" variables match the location of your files. Additionally, you can use the "tflite_filename" variable to choose a name for the .tflite file that will be created after the training process.

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/path_setting.png">
</p>

To choose the model you want to train, you need to modify the relevant section of the code. This can be found in the following lines:

    # Google colab ---> load_training_validation.ipynb
    spec = model_spec.get('efficientdet_lite2')
    
We used the efficientdet_lite2 model because it balances precision requirements with an acceptable training time. You can try other models by changing the number (0-4), but keep in mind that slower but more accurate models will require more resources. Additionally, you need to adjust the batch_size (number of photos trained at once) and epochs (number of times dataset is trained) according to your needs. A smaller batch_size will train photos faster, but larger batch sizes are more efficient with large datasets. The number of epochs depends on the amount of time you have and the desired accuracy of your model.
 
 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/training.jpg" | width=800 >
</p>
     
You can do this setting in the following lines of code: 

    # Google colab ---> load_training_validation.ipynb
    model = object_detector.create(training_data, model_spec=spec, epochs=120, batch_size=16, train_whole_model=True, validation_data=validation_data)
     
In our case, the best results are achieved with a number of epochs = 120 and a batch_size = 16. Note that a very high number of epochs can cause overfitting and make the model not generalize well and that a higher number of batch_sizes can improve model accuracy but will have a larger impact on google colab resources and may exceed the GPU memory limit.

There is one last detail, you must indicate the labels that you put in the LabelImage program. In this case our labels are: "white type" and "wild type". It can be modified in the following part of the code:

    # Google colab ---> load_training_validation.ipynb
    training_data = object_detector.DataLoader.from_pascal_voc(
        training_path, 
        training_path, 
        ["white type", "wild type"]
    )
    validation_data = object_detector.DataLoader.from_pascal_voc(
        validation_path, 
        validation_path, 
        ["white type", "wild type"]
    )

To run the code and get the trained model in a .tflite file, simply run each cell sequentially by clicking on the play icon to the left of each cell. It's important to not skip any cells or the code won't work and you'll need to restart and execute the cells again.

After training, the model is evaluated and compared before and after being exported to a .tflite model. You can see the results directly in the google colab notebook and ensure that the values are not significantly different. If they are, you may need to adjust your training and validation files and repeat the process.

Note that the mean average precision (mAP) was calculated using COCO (https://cocodataset.org/).

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/validation.jpg" | width=800 >
</p>

---

### 3.) IMAGE PROCESSING

#### 3.1) Image processing with Google colab option:

To use this option, you need a Gmail account and a specific folder structure. Create a folder called "Image_Processing_Approach" in your Google Drive with the image you want to process (default: "01.JPG") and the trained model (default: drosophila_lite2_epochs120_batch16_img1251_wild_white_v7_V2.tflite). Additionally, create a folder called "processed" inside "Image_Processing_Approach", which must be empty. However, Google Drive does not allow empty folders to be uploaded, so a code is included to create the folder in the Google Colab notebook. The code divides a photo into 20 parts and passes each fragment through the AI. The Google Colab link is: https://colab.research.google.com/drive/1yYFuL3nnxHVXfjSvsRz-pFrDTsFPO2Y7?usp=sharing.

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/image_processing.jpg" | width=800 >
</p>

To use Google Colab code, simply run the code cells sequentially.

As output, a .csv file similar to this will be obtained:

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/example_output.png">
</p>

You can also rebuild the original image with the processed images and download it:

 <p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/reconstructed_image.png">
</p>

---

### 4.) INTERNET IMAGE PROCESSING

Out of curiosity, we wanted to test what happens when we apply our trained model with images taken directly from the internet. For this, the "drosophila_counter.py" script was adapted so that it would not divide the photos into 20 pieces and would process them directly.

If you want to try it, download and move the "INTERNET IMAGE PROCESSING" folder to your desktop. You will need to make the changes mentioned in section 3.2 of this repository to be able to run the "drosophila_counter_one_photo.py" script locally. The image that is processed must be inside the folder where the script is located and it must be called "TEST_1.jpg" because it is the file that it will look for by default. Some of the results we have obtained are the following:

<p align="center">
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/processed_1.png" | width=400 >
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/processed_2.png" | width=400 >
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/processed_3.png" | width=400 >
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/processed_4.png" | width=400 >
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/processed_5.png" | width=400 >
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/processed_6.png" | width=400 >
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/processed_7.png" | width=400 >
  <img src="https://github.com/AaronGS1999/Drosophila_Tensorflow/blob/main/images/processed_8.png" | width=800 >
</p>

As you can see, the model is not so badly adapted to other conditions, even to drawings, but it will always work better for the type of photos with which it has been trained.

---
