#!/bin/bash

IMAGE_NAME="python-env"

echo "Building Docker image: $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

echo "Running Docker container from image: $IMAGE_NAME..."
docker run -it --rm -v $(pwd):/app $IMAGE_NAME