# Required libraries
import os
import errno
import time
import numpy as np
from numpy import RAISE
import cv2
from tflite_support import metadata
import tensorflow as tf
import platform
from typing import List, NamedTuple
import json
from PIL import Image
import pandas as pd
import image_slicer 
from pathlib import Path

Interpreter =tf.lite.Interpreter
load_delegate = tf.lite.experimental.load_delegate

# Required variables
cont = 1
bucle_while = True
wildtypecount = 0
whitetypecount = 0
imagewildtypecount = 0
imagewhitetypecount = 0
counter1 = 1
counter2 = 1
dataframe = pd.DataFrame(columns=["IMAGE_NAME", "WILD", "WHITE"])

# required classes
class ObjectDetectorOptions(NamedTuple):
  

  enable_edgetpu: bool = False
  

  label_allow_list: List[str] = None
  

  label_deny_list: List[str] = None
  

  max_results: int = -1
  

  num_threads: int = 1
  

  score_threshold: float = 0.0
  


class Rect(NamedTuple):
  left: float
  top: float
  right: float
  bottom: float


class Category(NamedTuple):
  label: str
  score: float
  index: int


class Detection(NamedTuple):
  bounding_box: Rect
  categories: List[Category]


def edgetpu_lib_name():
  return {
      'Darwin': 'libedgetpu.1.dylib',
      'Linux': 'libedgetpu.so.1',
      'Windows': 'edgetpu.dll',
  }.get(platform.system(), None)


class ObjectDetector:
  _OUTPUT_LOCATION_NAME = 'location'
  _OUTPUT_CATEGORY_NAME = 'category'
  _OUTPUT_SCORE_NAME = 'score'
  _OUTPUT_NUMBER_NAME = 'number of detections'

  def __init__(
      self,
      model_path: str,
      options: ObjectDetectorOptions = ObjectDetectorOptions()
  ) -> None:

    # Load metadata from model.
    displayer = metadata.MetadataDisplayer.with_model_file(model_path)

    # Save model metadata for preprocessing later.
    model_metadata = json.loads(displayer.get_metadata_json())
    process_units = model_metadata['subgraph_metadata'][0]['input_tensor_metadata'][0]['process_units']
    mean = 0.0
    std = 1.0
    for option in process_units:
      if option['options_type'] == 'NormalizationOptions':
        mean = option['options']['mean'][0]
        std = option['options']['std'][0]
    self._mean = mean
    self._std = std

    # Load label list from metadata.
    file_name = displayer.get_packed_associated_file_list()[0]
    label_map_file = displayer.get_associated_file_buffer(file_name).decode()
    label_list = list(filter(lambda x: len(x) > 0, label_map_file.splitlines()))
    self._label_list = label_list

    # Initialize TFLite model.
    if options.enable_edgetpu:
      if edgetpu_lib_name() is None:
        raise OSError("The current OS isn't supported by Coral EdgeTPU.")
      interpreter = Interpreter(
          model_path=model_path,
          experimental_delegates=[load_delegate(edgetpu_lib_name())],
          num_threads=options.num_threads)
    else:
      interpreter = Interpreter(
          model_path=model_path, num_threads=options.num_threads)

    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]

    sorted_output_indices = sorted(
        [output['index'] for output in interpreter.get_output_details()])
    self._output_indices = {
        self._OUTPUT_LOCATION_NAME: sorted_output_indices[0],
        self._OUTPUT_CATEGORY_NAME: sorted_output_indices[1],
        self._OUTPUT_SCORE_NAME: sorted_output_indices[2],
        self._OUTPUT_NUMBER_NAME: sorted_output_indices[3],
    }

    self._input_size = input_detail['shape'][2], input_detail['shape'][1]
    self._is_quantized_input = input_detail['dtype'] == np.uint8
    self._interpreter = interpreter
    self._options = options

  def detect(self, input_image: np.ndarray) -> List[Detection]:
    image_height, image_width, _ = input_image.shape

    input_tensor = self._preprocess(input_image)

    self._set_input_tensor(input_tensor)
    self._interpreter.invoke()

    # Get all output details
    boxes = self._get_output_tensor(self._OUTPUT_LOCATION_NAME)
    classes = self._get_output_tensor(self._OUTPUT_CATEGORY_NAME)
    scores = self._get_output_tensor(self._OUTPUT_SCORE_NAME)
    count = int(self._get_output_tensor(self._OUTPUT_NUMBER_NAME))

    return self._postprocess(boxes, classes, scores, count, image_width,
                             image_height)

  def _preprocess(self, input_image: np.ndarray) -> np.ndarray:

    # Resize the input
    input_tensor = cv2.resize(input_image, self._input_size)

    # Normalize the input if it's a float model (aka. not quantized)
    if not self._is_quantized_input:
      input_tensor = (np.float32(input_tensor) - self._mean) / self._std

    # Add batch dimension
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return input_tensor

  def _set_input_tensor(self, image):
    tensor_index = self._interpreter.get_input_details()[0]["index"]
    input_tensor = self._interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

  def _get_output_tensor(self, name):
    output_index = self._output_indices[name]
    tensor = np.squeeze(self._interpreter.get_tensor(output_index))
    return tensor

  def _postprocess(self, boxes: np.ndarray, classes: np.ndarray,
                   scores: np.ndarray, count: int, image_width: int,
                   image_height: int) -> List[Detection]:
    results = []

    # Parse the model output into a list of Detection entities.
    for i in range(count):
      if scores[i] >= self._options.score_threshold:
        y_min, x_min, y_max, x_max = boxes[i]
        bounding_box = Rect(
            top=int(y_min * image_height),
            left=int(x_min * image_width),
            bottom=int(y_max * image_height),
            right=int(x_max * image_width))
        class_id = int(classes[i])
        category = Category(
            score=scores[i],
            label=self._label_list[class_id],  # 0 is reserved for background
            index=class_id)
        result = Detection(bounding_box=bounding_box, categories=[category])
        results.append(result)

    # Sort detection results by score ascending
    sorted_results = sorted(
        results,
        key=lambda detection: detection.categories[0].score,
        reverse=True)

    # Filter out detections in deny list
    filtered_results = sorted_results
    if self._options.label_deny_list is not None:
      filtered_results = list(
          filter(
              lambda detection: detection.categories[0].label not in self.
              _options.label_deny_list, filtered_results))

    # Keep only detections in allow list
    if self._options.label_allow_list is not None:
      filtered_results = list(
          filter(
              lambda detection: detection.categories[0].label in self._options.
              label_allow_list, filtered_results))

    # Only return maximum of max_results detection.
    if self._options.max_results > 0:
      result_count = min(len(filtered_results), self._options.max_results)
      filtered_results = filtered_results[:result_count]

    return filtered_results


_MARGIN = 20  # pixels
_ROW_SIZE =20  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image: np.ndarray,
    detections: List[Detection],
) -> np.ndarray:
  
  for detection in detections:
    # Draw bounding_box
    start_point = detection.bounding_box.left, detection.bounding_box.top
    end_point = detection.bounding_box.right, detection.bounding_box.bottom
    cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    class_name = category.label

    # detection count
    if class_name == "wild type":
      global imagewildtypecount
      global wildtypecount
      wildtypecount = wildtypecount + 1
      imagewildtypecount = imagewildtypecount + 1
    if class_name == "white type":
      global whitetypecount
      global imagewhitetypecount
      whitetypecount = whitetypecount + 1
      imagewhitetypecount = imagewhitetypecount + 1
    probability = round(category.score, 2)
    result_text = class_name + ' (' + str(probability) + ')'
    text_location = (_MARGIN + detection.bounding_box.left,
                     _MARGIN + _ROW_SIZE + detection.bounding_box.top)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
  new_line = {"IMAGE_NAME":name_photo, "WILD":imagewildtypecount, "WHITE":imagewhitetypecount}
  global dataframe
  dataframe = dataframe.append(new_line, ignore_index=True) 
  imagewildtypecount = 0
  imagewhitetypecount = 0
  return image

path_folder = "C:/Users/aaron/Desktop/IMAGE PROCESSING"
path_folder2 = "C:/Users/aaron/Desktop/IMAGE PROCESSING/processed"
INPUT_IMAGENAME = "/01.JPG"
cont = 21
# fragment the image
image_slicer.slice(path_folder + INPUT_IMAGENAME, 20)
file_name = Path(path_folder + INPUT_IMAGENAME).stem
# Image processing
for i in range(1, cont ):
    name_photo = "01_0"+ str(counter2)+"_0" + str(counter1) + ".png"
    INPUT_IMAGE = path_folder + "/"+ name_photo
    DETECTION_THRESHOLD = 0.65
    TFLITE_MODEL_PATH = "C:/Users/aaron/Desktop/Troceador/drosophila_lite2_epochs120_batch16_img1251_wild_white_v7_V2.tflite"

    image = Image.open(INPUT_IMAGE).convert('RGB')
    #image.thumbnail((512, 512), Image.ANTIALIAS)
    image_np = np.asarray(image)
    # Load the TFLite model
    options = ObjectDetectorOptions(
        num_threads=4,
        score_threshold=DETECTION_THRESHOLD,
    )
    detector = ObjectDetector(model_path=TFLITE_MODEL_PATH, options=options)

    # Run object detection estimation using the model.
    detections = detector.detect(image_np)

    # Draw keypoints and edges on input image
    image_np = visualize(image_np, detections)
    # Save the detection result
    im = Image.fromarray(image_np)
    im.save(path_folder2 + "/" + name_photo)
    print("processed image "+str(i)+" of "+ str(cont - 1))
    counter1 = counter1 +1
    if counter1 == 6:
        counter2 = counter2 +1 
        counter1 = 1

# show results    
print("The total number of wild types detected has been: " + str(wildtypecount))
print("The total number of white type detected has been: "+ str(whitetypecount))

tiles = image_slicer.open_images_in(path_folder2)
image = image_slicer.join(tiles)
image.save(path_folder2+"/reconstructed_image.png")
# save results
dataframe.to_csv('drosophila_output.csv')
