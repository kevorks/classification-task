# Classification Task


## Project Overview
-------------------
Classification is a supervised learning method that is sufficient to derive a so-called class from the data. While in regression, the exact prediction is a continuous number, in classification, the prediction is a class, i.e., categorical data is sought.

In this project, I'll apply machine learning classification algorithms using the well-known Titanic Dataset, where the task is "What kind of people were likely to survive?" The algorithms to be used are ```RandomForest```, ```LightGBM``` and ```LogisticRegression```.

## Directory Setup
------------------
The folder structure with the relevant scripts and files is demonstrated below:

```markdown
    ├── data
    │    └── titanic.csv
    ├── figs
    │    └── ...
    ├── src
    │    ├── Dockerfile
    │    ├── logger.conf
    │    └── main.py
    ├── docker-compose.yml
    ├── README.md
    └── requirements.txt
```

## Descriptions 
---------------

* **data**: This folder contains the Titanic CSV dataset. 

 * **figs**: This folder contains the plots created in the ```main.py``` script.

* **src**: This folder contains the following:
    * **Dockerfile**: A Dockerfile to set up a Docker image with Python 3.10
    * **logger.conf**: A file that configures the logging settings for a Python application, specifying log message formatting, output handlers (in this case, console output), and the log level (INFO) for the root logger, ensuring that log messages are displayed with timestamps, log levels, and directed to the console.
    * **main.py**: The script that handles the whole functionality of the project.

* **docker-compose.yml**: A docker-compose.yml file that defines a service named classification-task-project, configures it to build an image using the Dockerfile located in the src directory, sets up volume mappings for data and source code directories, and specifies the command to run when starting the service, which is ```python3 main.py```.


## Execution without Docker
### Prerequisites
----------------

This project involves development using python `3.10` version. The necessary requirements needs to be installed to develop.

It is highly recommended setting up a virtual environment.

After setting up the virtual environment, you can install the requirements through following command:

```
pip install -r requirements.txt
```

### Execution
------------

To be able to run the script for training purposes, execute the following commands under the `src` folder:

```
python3 main.py
```

## Execution with Docker
------------------------

To simplify the deployment of the project, Docker containers can be used. Here's how to set up and run the project using Docker:

1. **Install Docker and Docker Compose**: Download and install Docker for your system by following the instructions on the [Docker website](https://www.docker.com/get-started).

2. **Clone the Repository**: Open a terminal and clone the repository:

   ```bash
   git clone https://github.com/kevorks/classification-task.git
   cd classification-task
   ```

3. **Build and Run the Docker Container**: In the root directory of the project, run the following command to build and start the Docker container defined in the `docker-compose.yml` file:

   ```bash
   docker-compose build
   docker-compose up
   ```

The commands above will create and start the container.

4. **Stop the Container**: To stop the container and remove it, run:

   ```bash
   docker-compose down
   ```

## Maintainers
--------------

* **Sevag Kevork** - *Maintainer/Data Scientist* - [me@sevagkevork.net](https://github.com/kevorks)

