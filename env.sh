docker run --rm -it --gpus all -v $(pwd)/..:/home/ --name InSpect kest3869/inspect bash -c "cd /home/InSpect && exec bash"
