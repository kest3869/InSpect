#!/bin/bash

# Use the first argument as the target directory, or default to current directory
TARGET_DIR=${1:-"./"}

# Set permissions to 777
sudo chmod -R 777 "$TARGET_DIR"

# Set ownership to nobody:nogroup
sudo chown -R nobody:nogroup "$TARGET_DIR"

# Print a confirmation message
sudo echo "Permissions and ownership have been reset to 777 and nobody:nogroup for $TARGET_DIR"