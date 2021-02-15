from scripts.image_saver import ImageSaver

"""
Create a dataset from the collected data in carla simulator. 
This will create a dataset folder, where images and their corresponding
labels are stored. It uses a specific folder structure to pass to the 
neural network of the Ultra Fast Lane Detection algorithm.
"""

if __name__ == '__main__':
    ImageSaver().execute()
    