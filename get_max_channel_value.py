from PIL import Image
import glob
import numpy
import os
from PIL import TiffImagePlugin
TiffImagePlugin.DEBUG = True
import tifffile
import cv2


print("Entered max value check")
print(os.getcwd())

max_0 = 0
max_1 = 0
max_2 = 0
max_3 = 0

for filename in glob.glob('./../../cdata_overlap/val/image/*.tif'):
  #print(filename)
  #print(filename[:2] + 'c' + filename[2:])
  #top one for images, bottom for labels with this dataset
  #tiff_file = cv2.convertScaleAbs(tifffile.imread(filename), alpha=(255.0/65535.0))
  tiff_file = cv2.convertScaleAbs(tifffile.imread(filename))
  #print(tiff_file)
  #print(tiff_file.shape)
  if (numpy.amax(tiff_file[0]) > max_0):
    max_0 = numpy.amax(tiff_file[0])
    print("New max_0: ", max_0)
  if (numpy.amax(tiff_file[1]) > max_1):
    max_1 = numpy.amax(tiff_file[1])
    print("New max_1: ", max_1)
  if (numpy.amax(tiff_file[2]) > max_2):
    max_2 = numpy.amax(tiff_file[2])
    print("New max_2: ", max_2)
  if (numpy.amax(tiff_file[3]) > max_3):
    max_3 = numpy.amax(tiff_file[3])
    print("New max_3: ", max_3)
  #print(numpy.amax(tiff_file[0]))
  #print(numpy.amax(tiff_file[1]))
  #print(numpy.amax(tiff_file[2]))
  #print(numpy.amax(tiff_file[3]))

