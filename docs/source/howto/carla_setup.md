

# CARLA setup
## Requirements
In order to generate a dataset in CARLA, you need to make sure, that CARLA simulator is installed. At this point it is very important to note, that it is recommended to install ```version 0.9.10```, since this was the build all the code was developed on. 

---
**NOTE**

You can also let it run on an older version of CARLA, but we can't promise that it will work. If you want to stay on the safe side, just install the version mentioned above.

---

##  Installation
CARLA Simulator can be installed using the following link: 
[https://github.com/carla-simulator/carla/blob/master/Docs/download.md](https://github.com/carla-simulator/carla/blob/master/Docs/download.md)

For more information on installation and builds refer to CARLA's official github respository.

For our project do the following:
1. Locate your CARLA installation folder and navigate to the ```PythonAPI``` directory:
	```Shell
	cd CARLA_0.9.10\WindowsNoEditor\PythonAPI
	```
2. Clone our project.
	```Shell
	git clone <git link>
	cd carla-lanedetection
	```
3. Install necessary dependencies.
	```Shell
	pip install -r requirements.txt
	```
4. Run the scripts described in [this Howto](generate_dataset_from_carla).
## Setup
Before you start creating your own dataset, make sure that your carla server is running. 
If the server is not running, you can do so by doing the following:

For Windows Users:
- If you prefer to use the GUI, go to your main directory and navigate to `\CARLA_0.9.10\WindowsNoEditor`. There you should find a file called `CarlaUE4.exe`. To start the server, just launch this file. **Note**: This way the server is launched without any parameters. All default settings are applied for the server. If you want to apply different parameters, make sure to use the following method.
- If you prefer the commandline, go to your main directory.
Navigate to `WindowsNoEditor` and launch the `CarlaUE4.exe`: 
    ```Shell
	cd CARLA_0.9.10\WindowsNoEditor
	
	# This way allows you to set parameters like windowsize and quality-level
	CarlaUE4.exe -carla-server -windowed -ResX=480 -ResY=320 -quality-level=Low -benchmark -fps=10
	```
For Linux Users:
- Navigate in your main directory, then run `CarlaUE4.sh` by entering the following command:
    ```Shell
	$ ./CarlaUE4.sh -carla-server -windowed -ResX=480 -ResY=320 -quality-level=Low -benchmark -fps=10
	```
After you launched the server, you can move on to the next step. 

**Hint**: While you are running your carla client scripts, the server should not be shut down. Otherwise the scripts send a timeout message and won't work.