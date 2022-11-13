
# PhysicsGym: A Reinforcement learning interface for Partial differential equation control
This repository contains the final submission code for Master's thesis: Developing a Reinforcement learning interface for partial differential equation control.


## Usage

Define the following methods specific to each problem

### 1. Initialisation
* ```obs_shape()``` : define the shape of your observation space
* ```action_shape()``` : define the shape of your action space
* ``` physics ``` : instantiate the physics object located in ```src/env/physics```

### 2. Reset
* ``` build_obs()``` : calculate the observation vector in this method.
* ``` build_reward() ``` : define reward calculation in this method.

### 3. Step
* ```action_transform()``` : define the transformation according to the problem, e.g. if you want to applying actions only at certain parts of the domain.
