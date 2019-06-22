#!/bin/bash

python -m scripts.main.calc_gait_params

python -m scripts.results.match_trials
python -m scripts.results.calc_error

python -m scripts.results.make_plots
