# Spatiotemporal gait measurement with a side-view depth sensor using human joint proposals.

[![build](https://github.com/ajhynes7/scikit-spatial/actions/workflows/main.yml/badge.svg)](https://github.com/ajhynes7/side-view-depth-gait/actions/workflows/main.yml)


This repository contains the code for the journal article *Spatiotemporal Gait Measurement with a Side-View Depth Sensor Using Human Joint Proposals, which is currently under review.


## Requirements

Python version 3.7 or higher is required. 

Some scripts for the results and figures make use of LaTeX for formatting. You can either install LaTeX or edit the code so it is not needed.


## Setup

### Creating a virtual environment

Create a virtual environment and activate it.

```bash
$ mkdir virtualenvs/
$ python -m venv virtualenvs/depth_gait_env

$ source virtualenvs/depth_gait_env/bin/activate
```

Finally, install the required Python packages into the environment.

```bash
$ cd depth-gait-analysis/
$ pip install -r requirements.txt
```


## Usage

### Method

The following scripts constitute the method of the paper.

Estimate the lengths of body links on each trial:
```bash
$ python -m scripts.main.estimate_lengths
```

Select the best proposals for head and foot positions:
```bash
$ python -m scripts.main.select_proposals
```

Cluster the frames of the walking trials to determine the walking passes:
```bash
$ python -m scripts.main.label_passes
```

Calculate gait parameters from the selected head and foot positions:
```bash
$ python -m scripts.main.calc_gait_params
```

For convenience, all of these scripts can be run at once:
```bash
$ python -m scripts.main.run_all_main
```


### Results

Run all of the results:
```bash
$ python -m scripts.results.run_all_results
```


### Figures

Generate all of the figures used in the paper:
```bash
$ python -m scripts.figures.run_all_figures
```
