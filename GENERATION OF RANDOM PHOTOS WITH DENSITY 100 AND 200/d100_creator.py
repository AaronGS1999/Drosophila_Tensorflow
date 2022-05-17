import os
import image_slicer
import splitfolders
import errno
import numpy as np
from numpy import RAISE
from PIL import Image
from shutil import rmtree

user = "aaron"
output_folder = "C:/Users/"+ user +"/Desktop/GENERATION OF RANDOM PHOTOS WITH DENSITY 100 AND 200/output/"

try:
    rmtree(output_folder)
except OSError as error:
    if error.errno != errno.EEXIST:
        RAISE
folder = "C:/Users/"+ user +"/Desktop/GENERATION OF RANDOM PHOTOS WITH DENSITY 100 AND 200/splitter/cuts"
splitfolders.ratio(folder, output="output",seed=np.random.random(), ratio=(.651, 0.349)) 

counter1 = 1
counter2 = 1
name = "d100"
carpeta = "C:/Users/"+ user +"/Desktop/GENERATION OF RANDOM PHOTOS WITH DENSITY 100 AND 200/output/train/cuts_D100/"

for filename in os.listdir(carpeta):
    if name in filename.lower():
        nombre_final ="_0"+ str(counter2)+"_0" + str(counter1) + ".png"
        print("Renamed {} as {}".format(filename, nombre_final))
        os.rename(carpeta + filename, carpeta + nombre_final)
        counter1 = counter1 +1
        if counter1 == 6:
            counter2 = counter2 +1 
            counter1 = 1

path_folder = carpeta
tiles = image_slicer.open_images_in(path_folder)
image = image_slicer.join(tiles)
image.save(path_folder+"reconstructed_image.png")    
im = Image.open("C:/Users/"+ user +"/Desktop/GENERATION OF RANDOM PHOTOS WITH DENSITY 100 AND 200/output/train/cuts_D100/reconstructed_image.png")
rgb_im = im.convert('RGB')
rgb_im.save("C:/Users/"+ user +"/Desktop/GENERATION OF RANDOM PHOTOS WITH DENSITY 100 AND 200/output/train/CUTS_D100/01.JPG", quality=100)    