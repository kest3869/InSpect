docker run -it --rm --gpus all -v $(pwd)/..:/home/ --name InSpect kest3869/inspect bash -c "cd /home/InSpect/ && exec bash"
