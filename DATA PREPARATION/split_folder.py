import splitfolders
folder = "C:/Users/aaron/Desktop/Tensorflow drosophila/divisor_fotos/dataset"
splitfolders.ratio(folder, output="output", seed=1999, ratio=(.8, 0.1, 0.1)) 