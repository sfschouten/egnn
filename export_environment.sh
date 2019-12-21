#!/bin/bash
conda env export | cut -f -2 -d "=" | grep -v "prefix" > environment.yml
