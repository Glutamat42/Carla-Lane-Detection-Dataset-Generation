
# ==============================================================================
# -- CARLA Variables ----------------------------------------------------------
# ==============================================================================

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FPS = 20
FOV = 90.0
CARLA_TOWN = 'Town03_Opt'

# Save images and labels on disk
isSaving = True
# Keep own vehicle either in center of road or oscillate between lanemarkings
isCenter = True
# Draw all lanes in carla simulator
draw3DLanes = False
# Calculate and draw 3D Lanes on Juction
junctionMode = True
# Third-person view for the ego vehicle
isThirdPerson = False

# Number of images stored in a .npy file
number_of_images = 100
# Total number of .npy files
total_number_of_imagesets = 100
# Vertical startposition of the lanepoints in the 2D-image
row_anchor_start = 160
# Number of images after agent is respawned
images_until_respawn = 350
# Distance between the calculated lanepoints
meters_per_frame = 1.0
# Total length of a lane_list
number_of_lanepoints = 80
# Max size of images per folder
max_files_per_classification = 2000

h_samples = []
for y in range(row_anchor_start, IMAGE_HEIGHT, 10):
	h_samples.append(y)


# ==============================================================================
# -- Saver Variables ----------------------------------------------------------
# ==============================================================================

# Output for .npy files
output_directory = 'data/rawimages/'
# Loading directory of .npy files
loading_directory = output_directory + CARLA_TOWN + '/'
# Path to the image and label files
saving_directory = 'data/dataset/' + CARLA_TOWN + '/'
train_gt = saving_directory + 'train_gt_tmp.json'
test_gt = saving_directory + 'test_gt.json'
overall_train_gt = saving_directory + 'train_gt.json'
