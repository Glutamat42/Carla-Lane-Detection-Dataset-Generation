import os
import json
import config as cfg

class LabelSaver():
    """
    Helper class to save all the lanedata (labels). Each label contains a list 
    of the x values of a lane, their corresponding predefined y-values and 
    their path to the image.
    """
    def __init__(self, label_file):
        self.image_name = 0
        
        folder = os.path.dirname(label_file)
        if not os.path.isdir(folder):
            os.makedirs(folder)
            
        if os.path.exists(label_file):
            os.remove(label_file)
        
        self.file = open(label_file, 'a')
        

    def add_label(self, x_lane_list):
        filestring = {"lanes": x_lane_list,
                      "h_samples": cfg.h_samples,
                      "raw_file": cfg.saving_directory + f'{self.image_name:04d}' + '.jpg'}
        
        jsonstring = json.dumps(filestring)
        self.file.write(jsonstring + '\n')
        self.image_name += 1
    
    
    def close_file(self):
        self.file.close()