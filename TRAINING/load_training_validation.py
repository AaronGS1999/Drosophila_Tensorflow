import numpy as np
import os
from colorama import Fore, init
init()

from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from tflite_support import metadata

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

"""
file path
The .xml files must be together with the .jpg files
"""

training_path = "" #Add the path of the training files
validation_path ="" #Add the path of the validation files
export_dir= "" # Add the path where the model will be exported
tflite_filename="drosophila_lite2_epochs200_batc2_img104_new_dataset.tflite" # You can change this to whatever name you want
model_dir = export_dir + tflite_filename

# Selecting the model architecture
spec = model_spec.get('efficientdet_lite2')

# data upload 
training_data = object_detector.DataLoader.from_pascal_voc(
    training_path, 
    training_path, 
    ['wildtype', 'whitetype'] # Change this to your labels
)
validation_data = object_detector.DataLoader.from_pascal_voc(
    validation_path, 
    validation_path, 
    ['wildtype', 'whitetype'] # Change this to your labels
)

# Training
model = object_detector.create(training_data, model_spec=spec, epochs=200, batch_size=2, train_whole_model=True, validation_data=validation_data)

# Model evaluation
print(Fore.GREEN +"") # Para destacar los resultados
print( model.evaluate(validation_data))
print(Fore.WHITE +"") # Recuperamos color normal

# Export model
model.export(export_dir=export_dir, tflite_filename=tflite_filename)

# Evaluation of the exported model
print(Fore.GREEN +"") # Para destacar los resultados
print(model.evaluate_tflite(model_dir, validation_data))
print(Fore.WHITE +"") # Recuperamos color normal
