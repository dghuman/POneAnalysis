#!/bin/bash
# The script that is called when submitting onto condor. Simply runs the python script for linefit and sets up the environment.

eval `/cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/setup.sh`
export PYTHONPATH="/home/users/ghuman/simAnalysis/reco:$PYTHONPATH"
i3env=/home/users/akatil/software/V06-01-02/build/env-shell.sh

$i3env /home/users/ghuman/_PYTHON/bin/python /home/users/ghuman/simAnalysis/reco/llhanalysis/likelihood.py -g pentagon10040 






