docker run -it --gpus all -v $(pwd)/..:/home/ --name InSpect kest3869/inspect bash -c "cd /home/Inspect && exec bash"
