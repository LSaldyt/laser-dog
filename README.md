# Spot-Multiagent 

This project focuses on reinforcement learning challenges for the "Rex" spot-micro quadruped robot.

The rex-gym package is somewhat old, and *requires* at most Python `3.7`. 
Also, tensorflow is currently pinned to `1.15.4`.

Full dependencies are specified with the [poetry](https://python-poetry.org) package in the `pyproject.toml` and `poetry.lock` files.


## Commands:  

First, start by downloading the repository:  
`git clone --recurse-submodules https://github.com/LSaldyt/spot-micro-reinforcement-learning`  
If the repository is already cloned, use `git submodule update --init --recursive` to update git submodules.

To install packages using [poetry](https://python-poetry.org):  
`pip install poetry` or `pip3 install poetry`  
`poetry update`  
`poetry install`  
`poetry run rex-gym policy --env walk`

Ideally, use the `pip` command associated with a `python3.7` install.  
Otherwise, you will have to use `poetry env use /full/path/to/python` and point to your `python3.7` binary.  
On linux, this path is typically `/usr/bin/python3.7`, so the full command to get setup is `poetry env use /usr/bin/python3.7`.  

For example, installing `python3.7` specifically might look like this on Ubuntu:  
`sudo apt update`  
`sudo apt install python3.7`  
After completing this, the `/usr/bin/python3.7` binary should be present.

## What to run:

See the [rex-gym](https://github.com/nicrusso7/rex-gym/blob/master/README.md) instructions for full details.  
For example, use `poetry run rex-gym policy --env gallop` to run a pre-trained gallop policy.  
Use `poetry run rex-gym train --playground True --env gallop --log-dir logs` to run a GUI-based training session.  
To run a faster headless session, use `poetry run rex-gym train --env gallop --log-dir logs`  

### Possible environments:

Rex-gym starts with the following environments: `gallop`, `walk`, `turn`, `standup`, `go`, `poses`.

