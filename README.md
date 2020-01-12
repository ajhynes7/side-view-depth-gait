
---
Gait Analysis with a Side-View Depth Sensor
---

![build](https://travis-ci.org/ajhynes7/depth-gait-analysis)


This repository contains the code for the journal article *Gait Analysis with a Side-View Depth Sensor using Human Joint Proposals*, which is currently under review.


# Setup

## Creating a virtual environment

First, install `virtualenv`.

```bash
$ pip install virtualenv
```

Then create a virtual environment and activate it.

```bash
$ mkdir virtualenvs/
$ virtualenv virtualenvs/depth_gait_env

$ source virtualenvs/depth_gait_env/bin/activate
```

Finally, install the required Python packages into the environment.

```bash
$ cd depth-gait-analysis/
$ pip install -r requirements.txt
```
