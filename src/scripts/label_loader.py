import json
import config as cfg

class LabelLoader():
    """
    Helper class to load every lanepoint from the labelfiles. Extracts a list
    of 4 lanes containing their lanepoints as tuples (x,y).
    """
    def __init__(self, file, file2 = None):
        self.label_file = open(file, 'r')
        if file2 != None:
            self.label_file2 = open(file2, 'a')
        self.actual_line = ""


    def update_rawfile_in(self, new_value):
        try:
            line = self.actual_line
            label = json.loads(line)

            label['raw_file'] = new_value
            
            self.label_file2.write(json.dumps(label) + '\n')
            
        except ValueError:
            print('JSON Decoding failed. A JSON-object might be None')
            return


    def calculate_folder(self, y):
        """
        Determines the category of a label and the corresponding path. 
        categories are: 
            * location of eventually existing lanes 
                * Both: on both sides of the actual lane are lanemarkings as well as on the left and right lane
                * Left: on both sides of the actual lane are lanemarkings as well as on the left but there is no lane(-marking) at the right lane
                * Right: on both sides of the actual lane are lanemarkings as well as on the right but there is no lane(-marking) at the left lane
                * None: one or both lanemarkings of thee actual lane aren't existing
            * curves (calculate deviation of a straight line)
                * straight: the lane and its lanemarkings are straight
                * left_curve: the lane is more or less turning to the left
                * right_curve: the lane is more or less turning to the right

        Returns:
            The categories as string in folder structure.
        """
        # try:
        lanes = [[],[],[],[]]
        self.actual_line = self.label_file.readline()
        label = json.loads(self.actual_line)

        number_of_lanes = 0
        for i, x in enumerate(label['lanes']):
            lanes[i] = list(zip(x, y))
            lanes[i] = list(filter(lambda x: x[0] != -2, lanes[i]))
            if len(lanes[i]) > 1:
                number_of_lanes += 1

        lane_side = ''
        if lanes[2] and lanes[3]:
            lane_side = 'Both'
        elif lanes[2] and not lanes[3]:
            lane_side = 'Left'
        elif not lanes[2] and lanes[3]:
            lane_side = 'Right'
        else:
            lane_side = 'None'

        selected_lane = 0
        if not lanes[selected_lane] or not lanes[selected_lane + 1]:
            return 'nolanes/'

        compare_point_index = int(0.5 * len(lanes[selected_lane]))
        x0 = int(list(lanes[selected_lane][compare_point_index])[0])
        y0 = list(lanes[selected_lane][compare_point_index])[1]

        x1 = int(list(lanes[selected_lane][0])[0])
        y1 = list(lanes[selected_lane][0])[1]
        x2 = int(list(lanes[selected_lane][-1])[0])
        y2 = list(lanes[selected_lane][-1])[1]

        if (y2 - y1) == 0:
            return 'nolanes/'
            
        # Herleitung über geradengleichung in parameterform
        straight_line_x = x1 + ((y0 - y1) / (y2 - y1)) * (x2 - x1)
        if straight_line_x - x0 < -2.5:
            return lane_side + '/left_curve/'  
        elif straight_line_x - x0 > 2.5:
            return lane_side + '/right_curve/'
        else:
            return lane_side + '/straight/'
        
            
        # except ValueError:
        #     print('JSON Decoding failed. A JSON-object might be Nonegere')
        #     return


    def load_lanepoints(self):
        """
        Loads every lanepoint from the labelfiles.
        Extracts a list of 4 lanes, which contain their
        lanepoints as tuples (x,y)

        Returns
        -------
        lanes : list
            List of 4 lanes.
        """
        try:
            y = cfg.h_samples
            lanes = [[],[],[],[]]
            line = self.label_file.readline()
            label = json.loads(line)

            for i, x in enumerate(label['lanes']):
                lanes[i] = list(zip(x, y))
                
        except ValueError:
            print('JSON Decoding failed. A JSON-object might be None')
            return

        return lanes
        

    def close_file(self):
        self.label_file.close()
        if hasattr(self, 'label_file2'):
            self.label_file2.close()     
