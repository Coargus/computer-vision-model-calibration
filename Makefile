# Makefile
# IMPORTANT: Please find the right cuda dev container for your environment
SHELL := /bin/bash

install:
	pip install cvias@git+https://github.com/Coargus/computer-vision-inference-and-server.git@0.0.1
	pip install -e .