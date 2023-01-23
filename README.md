# Training set of Message Passing Network for Multi object tracking
This repository is a training set of Message Passing Network for Multi Object Tracking.
The idea of MPN for MOT comes from [Learning a Neural Solver for Multiple Object Tracking](https://github.com/dvl-tum/mot_neural_solver)

This program tracks points that gradually change color and move on a random walk.
Feature embedding is replaced by RGB colors and does not take into account the width and height of the detection window.

# Setup
You can set up an environment with Docker

```shell
docker build . --tag=tsmpn4mot
docker run --rm -ti --gpus all -v $PWD:/work -w /work tsmpn4mot bash
```

# Generates dataset

``` shell
sh data/generate.sh
```

# Run

``` shell
python main.py
```