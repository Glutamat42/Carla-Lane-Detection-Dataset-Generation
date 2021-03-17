
# HOWTO: Generate labeled data from CARLA
This HOWTO explains how you can collect image files with its corresponding labels in CARLA Simulator to later train a deep neural network to detect lanemarkings on the road.

## Introduction
At first it is necessary to understand the main folderstructure. The following tree structure shows how the core structure looks like:
    
		/carla-lanedetection
		  /src
		  ├── scripts
		  │   ├── buffered_saver.py
		  │   ├── image_saver.py
		  │   ├── label_loader.py
		  │   ├── label_saver.py
		  ├── config.py
		  ├── dataset_generator.py
		  ├── fast_lane_detection.py
		  ├── image_to_video.py
	
These are all the files that are needed to collect data from CARLA Simulator and convert them to a dataset. 
The bottom three files are the main files. These are the ones to execute in the commandline to start the whole process.
The scripts folder contains files that are included in the main files. They implement helper methods to (pre-)process all the data. 

Important steps of this process are:
1. Execute and run `fast_lane_detection.py`. 
This collects all the data in CARLA and saves them as .npy (numpy) files and generates temporary label files, which are later filtered. 

2. Execute and run `dataset_generator.py`. 
All the raw images and labels are then converted to .jpg images and .json labels. This file creates a `dataset` directory and places the processed files inside. Another task of this script is to balance the data.

3. Execute and run `image_to_video.py`.  (Optionally)
After generating the dataset, the images can then be converted to a video. This might be helpful, if you want to check your images and labels for errors.


## 1. Collect rawimages from CARLA
The script `fast_lane_detection.py` contains multiple classes, which are listed as follows:

#### CarlaSyncMode
The synchronous mode ensures, that if we have different sensors mounted on our ego vehicle or in the world, all of these sensors gather data at the exact same time. This is an important step, since we have two different cameras mounted on our ego vehicle, on the one hand we have an RGB camera sensor, that collects images from the world, and on the other hand we have a semantic segmentation camera, which is needed for filtering lanepoints on objects in the world (more on that later). It's basically really useful, when synchrony between different sensors is needed. In this mode, the server sets the speed of the overall simulation. All the data, which is coming from the client is put into a queue, and the server handles all these requests from the client. A really notably drawback from this is that the framerate drastically drops from 250+ FPS to about 20 FPS on a system with mediocre hardware (Intel i5 3240, NVidia 1060GTX, 8GB RAM). The class contains the communication and initializes the fixed time-steps and the behavior between server and client. For more information please refer to the official CARLA documentation website.

#### VehicleManager
This class manages the movement of the ego vehicle, as well as the spawning, despawning and movement of the other neighbor vehicles in the vicinity of the own car. The ego vehicle moves slightly different compared to the other cars. The oscillation (moving the vehicle from the left to the right between the lane markings) includes a zick-zack function and a rotation at the yaw-angle. Apart from the oscillation on the road, the ego vehicle uses a special system to move within the world. We decided to use a reliable system to move the own car precisely and quickly through the world. For this purpose, waypoints were an important tool. The algorithm of the movement of the ego vehicle is described as follows:

- Imagine we have a 2 dimenstional `waypoint_list`: 4 lanes and in each lane we have n waypoints. On startup we have to initialize our`waypoint_list` with n waypoints, the  `number_of_lanepoints` of the list can be adjusted in the `config.py.` It's set to 50, but it can be 100 of course, which means that more lanepoints are displayed in front of the car. It's basically a variable that determines how far we look on the road. The distance between the lanepoints can also be set, but for our purpose we set it to 1.0, which means that we look 50 meters ahead on our road, `meters_per_frame` can also be set to 0.5, so the lanepoints are just 0.5 meters apart from each other.
- Each time the server ticks, the ego vehicle is moved to the next waypoint within this list. You can also say that on each new frame the vehicle moves forward. Since we have initialized our `waypoint_list` with n waypoints, let's say 50, we only have to calculate a maximum of 200 lanepoints one single time on startup: 50 calculations per lanepoint and 4 lanes in total. As soon as we're in the gameloop, the computation gets reduced. On each tick, our ego vehicle has to look out for a new future lanepoint per lane and thus, we only have to compute a maximum of 4 lanepoints, since they are appended to the `waypoint_list`. So on each frame, new lanepoints are added and the old ones, which are behind the car, are removed from the list.
	
Another really basic concept/algorithm was how to create a realistic traffic scenario with other vehicles in the vicinity of the own agent. We came up with a concept, which randomizes a lot of things, e.g. how many cars to spawn or on which position. For this purpose, a method was created to spawn 5 vehicles first. After that, the vehicles' speed have to be locked to the speed of the own agent, so it looks like the neighbor vehicles are attached to the agent vehicle. Every`frame_counter` frames, the neighbor vehicles start to mix up and randomly spawn on different positions around the own car. 

Last but not least, there is a function, which is needed to detect, if a junction is ahead of our agent. This was useful to filter out junctions, we didn't want to have in our dataset. The deep neural network might not be able to learn junctions correctly and it might lead to missclassification of the lanes. 

#### LaneMarkings
This class is the most important one for lane detection. It contains functions for extracting and calculating the lane data from Carla and convert it to the correct format. The most important methods are described as follows:

`calculate3DLanepoints()`:
Uses the up-vector, the forward vector of the actual lane and the position of the waypoint(in the middle of a lane) to calculate where the lane should be. This information is calculated by the cross product of the forward- and up-vector. The resulting vector is normalized and multiplied by the half of the lane width to get the correct length. Now the result is added to the waypoint and the coordinates of the laneposition is reached. In addition to the Lanemarking position of the own lane the function does also calculate the position of respectively adjacent lanes if existing and not an a junction. After checking if there is a lanemarking at all (carla.LaneMarkingType) the function appends the 3D positions to the lanes list and returns them.

`calculate2DLanepoints()`:
Takes 3D-points of the Carla environment and transforms them to 2D-screen coordinates the camera recorded. Now the coordinates have to be formatted, so it can be passed into the neural network later. The wanted format are the x-values to a fix array of y-values for every lane. 

`calculateYintersections()`:
Calculates the intersection between the lanes and the horizontal row anchor lines, which are predefined by the deep neural network algorithm. This is done by connecting the 2D-coordinates and calculating their intersection with the horizontal the y-value line.

`filter2DLanepoints()`:
Since the calculated lanepoints could be behind a house, wall, etc. and aren't visible from the camera point of view, the deep neural network algorithm shouldn't get them as input since it would increase errors. These points are filtered out with the help of the sematic segmentation camera which must have the same Transformation as the camera.

On the top of that, there are several small functions that e.g. draw the calculated lanemarkings on the screen on runtime.

#### CarlaGame
This is our so-called main class of the script and it provides the fundamental functions that are necessary to run the game client. Upon running the script, the first function it calls is `execute()`. 

`initialize()`:
This is the first method being called from `execute()`, which is used to set some initial parameters. The initialization includes starting the synchronous mode with the carla server, loading the map for the simulation from the `config.py`, defining positions of the camera in relation to the actor, setting the actor and the cameras (RGB camera as well as semantic segmentation camera), setting the VehicleManager, putting the actor in the simulation and initialize the buffered saver to be ready to recieve data. 

`on_gameloop()`:
After that, this method is being called. It determines the control flow of every frame and handles different things:
- the movement of all vehicles
- taking an image and saving them using `buffered_saver.py`
- calculating the lane data and saving it using `labelsaver.py`
- rendering the display and drawing the calculated lanes on the screen
- avoiding that the car always drives on the same lane.
- after how many images a reset has to happen.    

##  2. Generate a dataset from the collected rawimages
We already covered the process of how to collect data in CARLA. Now in this step, we'll be covering the process of generating a dataset from the collected data. 

After executing the first step, you should now have a folder, where all the collected images are saved and an intermediate json file, which contains the lanemarking labels for each image. Executing `dataset_generator.py` calls a helperclass, which loads the numpy arrays. After loading it executes other methods which are described as follows: 

- `save_image_to_disk()`: Loads the labels to the corresponding image from the intermediate json file and converts the numpy array into .jpg images. 
- `calculate folder()`: Classifies a label according to additional lanemarkings apart from the actual lane and by curves or straight lines. On this basis it returns a folder in which the image belongs. According to this classification, the image is moved to the folder, but only if the amount of files has not yet exceeded `maximal_files_per_classification`, specified in `config.py`. The label gets then adapted by the new directory and saved in the final label file. 

It's not possible for the script to get images out of a numpy array and to balance the data properly, so that every specified classfolder has the same number of images. It's probably best to take a last look on each pair of data and filter out incorrect images and labels. For more detail, refer to step 3 of this documentation.

---
**NOTE**

When generating a dataset, you probably want to record your data images on different towns. If you want to use a different map, you have to navigate to `config.py` and rename `CARLA_TOWN` to e.g. 'Town03', 'Town04' or 'Town05'. Make sure that, after you've collected your data, you only have to execute the script `dataset_generator.py` once.

---

## 3. Convert the images to video (Optionally)

After generating a dataset, as described in the previous step, it's probably not a bad idea to check, if all the images have the right labels. For this purpose we created this script. It can be used to gather all the .jpg images, which are saved in `data/debug`, and convert them to a .mp4 video. This displays all the labels on the images. With this script, it's really easy to track down errors in the dataset. This was mainly used for debugging purposes in the development stage, but it's also suitable for just checking the dataset.

---
**NOTE**

When executing the script `image_to_video.py`, make sure that you enter the correct town in `config.py`. Unfortunately, you have to rerun the script for every town, that is used in your dataset. We might fix this in the future!

---