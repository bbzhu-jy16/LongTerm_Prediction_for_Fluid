# Learning High-Performance Long-term Temporal Evolution for Fluid Flow

### Install Mantaflow first
This installation guide focusses on Ubuntu 14.04 as a distribution. The process will however look very similar with other distributions, the main differences being the package manager and library package names.

First, install a few pre-requisites:

`sudo apt-get install cmake g++ git python3-dev qt5-qmake qt5-default`

Then, change to the directory to install the source code in, and obtain the current sources from the repository (or alternatively download and extract a source code package)

`cd <gitdir>`

To build the project using CMake, set up a build directory and choose the build options (explanation)

`mkdir mantaflow/build`

`cd mantaflow/build`

`cmake .. -DGUI=ON -DOPENMP=ON -DNUMPY=ON -DPYTHON_VERSION=3.6`

`make -j4`

That's it! You can now test mantaflow using an example scene

`./manta ../scenes/examples/simpleplume.py`

Common Linux problems:

- In conjunction with tensorflow, it can easily happen these days that you have multiple version of python installed. If cmake for mantaflow finds a different one than you're using for tensorflow, you will get errors such as ''ImportError: no module named XXX''. To fix this problem, manually select a python version in cmake with -DPYTHON_VERSION=X.Y
- It is also possible to directly specify the directory in which the python installation resides as follows:
    - `DPYTHON_INCLUDE_DIR=/PATH/TO/PYTHONENV/include/python3.6m `
    - `DPYTHON_LIBRARY=/PATH/TO/PYTHONENV/lib/libpython3.6m.so`

Further information on the installation process can be found on the project website http://mantaflow.com/.

## Model Training

Open the source directory
`cd <gitdir>/src`

2D Long-term Prediction model for density can be trained with

`python DensityPrediction2D.py -m train -r 64 --density_train_path path_to_density_data --velocity_train_path path_to_velocity_data`

2D Long-term Prediction model for velocity can be trained with

`python VelocityPrediction2D.py -m train -r 64 --velocity_train_path path_to_velocity_data`

3D Long-term Prediction model for density can be trained with

`python DensityPrediction3D.py -m train -r 64 --density_train_path path_to_density_data --velocity_train_path path_to_velocity_data`

3D Long-term Prediction model for velocity can be trained with

`python VelocityPrediction3D.py -m train -r 64 --velocity_train_path path_to_velocity_data`

## Model Test

2D Long-term Prediction model for density can be tested with

`python DensityPrediction2D.py -m test -r 64 --density_test_path path_to_density_data --velocity_test_path path_to_velocity_data --model_path path_to_model --model_name model_name`

2D Long-term Prediction model for velocity can be tested with

`python VelocityPrediction2D.py -m test -r 64 --velocity_test_path path_to_velocity_data --model_path path_to_model --model_name model_name`

3D Long-term Prediction model for density can be tested with

`python DensityPrediction3D.py -m test -r 64 --density_test_path path_to_density_data --velocity_test_path path_to_velocity_data --model_path path_to_model --model_name model_name`

3D Long-term Prediction model for velocity can be tested with

`python VelocityPrediction3D.py -m test -r 64 --velocity_test_path path_to_velocity_data --model_path path_to_model --model_name model_name`

## File path:
- Datasets: `<gitdir>/datasets/`
- Source Code Path: `<gitdir>/src/`
- Density Model Path: `<gitdir>/density_model/`
- Velocity Model Path: `<gitdir>/velocity_model/`
- Prediction Path： `<gitdir>/predictions/`

