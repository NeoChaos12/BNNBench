[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/automl/pybnn/blob/master/LICENSE)

# BNNBench
Bayesian Neural Network Benchmarker for Bayesian Optimization.

This repository contains a flexible framework for benchmarking the performance 
of BNNs as surrogate models for Bayesian Optimization.
The main modules are:
 - models: A collection of BNNs. See below for a list of available models.
 - emukit_interfaces: A collection of wrappers and helper modules to run the 
   BO benchmarking with the help of emukit.
 - postprocessing: Data collection, collation and post-processing module. 
   All collected data will be stored as Pandas dataframes.
 - visualization: Data visualization modules, designed for (almost) 
   arbitrarily structured Pandas DataFrames.

The following models have been implemented:
 - [MC-DropOut](http://mlg.eng.cam.ac.uk/yarin/PDFs/Dropout_as_a_Bayesian_approximation.pdf) (MCDO)
 - [MC-BatchNorm](https://arxiv.org/pdf/1802.06455v2.pdf) (MCBN)
 - [Deep Ensemble](https://papers.nips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf)
 - [Scalable Bayesian Optimization Using Deep Neural Networks](https://arxiv.org/pdf/1502.05700.pdf) (DNGO) - 
   Adapted from [this](https://github.com/automl/pybnn) implementation of DNGO.

# Installation

BNNBench can be installed by typing the following series of commands in a terminal on linux:

    git clone https://github.com/NeoChaos12/BNNBench.git
    cd bnnbench
    python setup.py install

Alternatively, after downloading the repo in a directory and ensuring all the 
dependencies are installed, set the environment variable "BNNBENCHPATH" to the 
full-path of the directory where the code was downloaded. Most of the useful 
run-scripts will detect this path and still work without installing the 
repository.  
