#!/bin/bash

# Default mode is build
mode="nothing"

# Parse command-line options
while getopts "br" opt; do
  case $opt in
    b)
      mode="build"
      ;;
    r)
      mode="run"
      ;;
    *)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Check if the chosen mode is supported
if [ "$mode" != "build" ] && [ "$mode" != "run" ]; then
  echo "Unsupported mode use ['-b' / '-r']" >&2
  exit 1
fi

# Execute the corresponding docker command based on the chosen mode
if [ "$mode" == "build" ]; then
  # Build the Docker image
  docker build -t sam:trt .
  # Execute the build command
  # ...
elif [ "$mode" == "run" ]; then
  # Run the Docker container
  xhost +
  docker run --gpus all -it --rm --net=host -e DISPLAY=$DISPLAY -v $(pwd):/workspace sam:trt
  # Execute the run command
  # ...
fi

