# InSpect v2.0
Classifying SPECT scans of psychiatric pathologies using deep learning techniques. 

## Project Overview
Development will use python instead of jupyter notebooks, have a torch backend instead of keras, and be done inside of docker containers. This will provide a more portable and scalable platform for the long term development of the InSpect project. 

## Instructions 
link to docker container: https://hub.docker.com/repository/docker/kest3869/inspect/general
docker pull command: docker pull kest3869/inspect:latest
running docker container: the script env.sh will mount the docker container in the expected structure, the data can be in this directory it's parent directory so that it remains in the scope of the docker container. The path to the data must be updated accordingly [INSERT GRAPHIC OF DATA STRUCTURE]

## Technical Updates
- published research approaches will be re-created and benchmarked on the problem
- more care will be taken with initial data loading and labelling 
- data will never be mixed and only clean data will be used for model fitting
- data will be stored as a pre-processed torch dataset 
- 2d approaches will be explored for better training speed, model diversity, and pre-trained model availability
- advanced data-augmentation techniques will be introduced to prevent over-fitting and balance dataset

## Legacy 
Contains first iteration of project coded in jupyter notebooks with a keras backend. 

### Author Details 
Kevin Stull 

CU Boulder Undergraduate in Applied Mathematics 

kest3869@colorado.edu

