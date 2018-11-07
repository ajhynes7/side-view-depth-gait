#!/bin/bash

coverage run -a kinect_lengths.py
coverage run -a kinect_best_pos.py
coverage run -a kinect_gait_parameters.py

coverage report
coverage html
