#!/bin/bash

python -m scripts.results.compare_positions
python -m scripts.results.compare_stances

python -m scripts.results.match_trials
python -m scripts.results.calc_error
python -m scripts.results.make_plots
