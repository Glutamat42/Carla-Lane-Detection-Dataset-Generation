import cv2
import os
import config as cfg

"""
Convert .jpg images to a .mp4 video file.
This was mainly used for debugging purposes.
Note that you must enter the correct path to the images you want to convert.
"""

loading_directory = 'data/debug/' + cfg.CARLA_TOWN
ext = '.jpg'
output = 'data/debug/' + cfg.CARLA_TOWN + '.mp4' # Save output file in lanedetection/data/

images = []
for filename in os.listdir(loading_directory):
    if filename.endswith(ext):
        images.append(filename)

# Determine the width and height from the first image
image_path = os.path.join(loading_directory, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video', frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

for image in images:
    image_path = os.path.join(loading_directory, image)
    frame = cv2.imread(image_path)
    out.write(frame)
    cv2.imshow('video',frame)
    
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

out.release()
cv2.destroyAllWindows()