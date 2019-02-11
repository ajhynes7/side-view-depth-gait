#!/bin/bash

coverage run -a -m scripts.pre_processing.process_kinect
coverage run -a -m scripts.pre_processing.process_zeno

coverage run -a -m scripts.main.run_all_main
coverage run -a -m scripts.results.run_all_results
coverage run -a -m scripts.figures.run_all_figures

coverage report -i
coverage html -i
