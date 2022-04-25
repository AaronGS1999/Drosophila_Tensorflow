import splitfolders
folder = "" # Add the address of your folders, keep in mind that they must have a specific structure of the library
splitfolders.ratio(folder, output="output", seed=1999, ratio=(.8, 0.1, 0.1)) 
