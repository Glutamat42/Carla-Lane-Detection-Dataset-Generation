import cv2
import numpy as np
import os
from PIL import Image
from scripts.label_loader import LabelLoader
import config as cfg

loading_directory = cfg.loading_directory
saving_directory = cfg.saving_directory


class ImageSaver():
    """
    Class used to load all the .npy files with LabelLoader and then convert into an image.
    Also draws the corresponding ground truth lanepoints on the images and saves the images seperately.
    """

    def execute(self):
        self.colormap = [(0, 255, 0),
                        (255, 0, 0),
                        (255, 255, 0),
                        (0, 0, 255)]

        self.label_loader = LabelLoader(cfg.train_gt)
        self.label_loader2 = LabelLoader(cfg.train_gt, cfg.overall_train_gt)
        self.image_name = 0
        self.image_name_gt = 0
        
        for i, image_set in enumerate(self.load_imagesets()):
            if(image_set.shape[0] == cfg.number_of_images):
                for image in image_set:
                    self.save_images_with_lanepoints(image)
                    self.save_image_to_disk(image)
                print('Saving imageset', i)
        
        self.label_loader.close_file()
        self.label_loader2.close_file()

    def load_imagesets(self):
        """
        Loads all the .npy files from the specified directory and appends these files to a list. 
        An Imageset is a .npy file, an imageset_list is a list of those imagesets.

        Returns:
            imageset_list: a list of imagesets (numpy arrays)
        """
        
        imageset_list = []
        for filename in os.listdir(loading_directory):
            if filename.endswith('.npy'):
                load_file = os.path.join(loading_directory, filename)
                imageset = np.load(load_file)
                imageset_list.append(imageset)
                print('Loading', load_file)
                
        return imageset_list
    

    def save_image_to_disk(self, image_array):
        """
        Converts the numpy array into .jpg images and saves it to a specific directory.
        Changes the label according to its calculated category and appends the label to the final json-file if the category is not already 
        full (defined by max_files_per_classification). This balances the input data for the algorithm
        In the actual implementation a image with the category nolanes wont be saved
        
        Args:
            image_array: numpy array
        """
        
        image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB), 'RGB')
        
        save_name = saving_directory + str(self.label_loader2.calculate_folder(cfg.h_samples))
        if not "nolanes" in save_name:
            folder = os.path.dirname(save_name)
            if not os.path.isdir(folder):
                os.makedirs(folder)
            files_in_folder = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
            if files_in_folder < cfg.max_files_per_classification:
                save_name = save_name + f'{(files_in_folder + 1) :04d}' + '.jpg'
                #image.save(save_name + '.jpg', 'JPEG')

                self.label_loader2.update_rawfile_in(save_name)
                image.save(save_name, 'JPEG')


    def save_images_with_lanepoints(self, image_array):
        """
        Saves the images with its corresponding ground truth labels as .jpg images to a specific directory.
        
        Args:
            image_array: numpy array.
        """
        
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        for i, lane in enumerate(self.label_loader.load_lanepoints()):
            for lanepoint in lane:
                cv2.circle(image_rgb, (lanepoint[0], lanepoint[1]), 3, self.colormap[i], thickness=1)
        
        image = Image.fromarray(image_rgb)
        save_name = os.path.join('data/debug/', f'{cfg.CARLA_TOWN}/{self.image_name_gt:04d}')
        self.image_name_gt += 1
        
        folder = os.path.dirname(save_name)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        
        image.save(save_name + '.jpg', 'JPEG')
