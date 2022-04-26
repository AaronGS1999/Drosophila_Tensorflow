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

training_path = "C:/Users/aaron/Desktop/Tensorflow drosophila/fotos_drosophila/entrenamiento"
validation_path ="C:/Users/aaron/Desktop/Tensorflow drosophila/fotos_drosophila/validacion"
export_dir= "C:/Users/aaron/Desktop"
tflite_filename="drosophila_lite2_epochs200_batc2_img104_new_dataset.tflite"
model_dir = "C:/Users/aaron/Desktop/" + tflite_filename

# Selecting the model architecture

"""
| Model architecture | Size(MB)* | Latency(ms)** | Average Precision*** |
|--------------------|-----------|---------------|----------------------|
| EfficientDet-Lite0 | 4.4       | 146           | 25.69%               |
| EfficientDet-Lite1 | 5.8       | 259           | 30.55%               |
| EfficientDet-Lite2 | 7.2       | 396           | 33.97%               |
| EfficientDet-Lite3 | 11.4      | 716           | 37.70%               |
| EfficientDet-Lite4 | 19.9      | 1886          | 41.96%               |

* Size of the integer quantized models.
** Latency measured on Raspberry Pi 4 using 4 threads on CPU.
*** Average Precision is the mAP (mean Average Precision) on the COCO 2017 validation dataset.
"""


spec = model_spec.get('efficientdet_lite2')

# data upload 
training_data = object_detector.DataLoader.from_pascal_voc(
    training_path, 
    training_path, 
    ['wildtype', 'whitetype']
)
validation_data = object_detector.DataLoader.from_pascal_voc(
    validation_path, 
    validation_path, 
    ['wildtype', 'whitetype']
)

# Training
"""
batch_size: Numero de datos que toma para entrenar la red cada vez
epochs: Numero de veces que pasa por el conjunto de datos de entrenamiento
"""
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