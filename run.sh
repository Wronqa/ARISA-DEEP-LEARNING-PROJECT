#!/bin/sh

docker build -t python-app .
docker run -it python-app bash