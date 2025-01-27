# CortexModel

## Overview
A rate-based model of adaptation in layer 2/3 of mouse V1

## Repo Contents

  - [Naive_opto_tails_prev_FBall_FBnotail_weightScalingTestTest.ipynb](./Naive_opto_tails_prev_FBall_FBnotail_weightScalingTestTest.ipynb) - Naїve model fitting and optogenetic manipulation
  - [DEP_sigmoid_unfixed_with_heatmap_fixedSM.ipynb](./DEP_sigmoid_unfixed_with_heatmap_fixedSM.ipynb) - Depressors fitting
  - [NA_init.ipynb](./NA_init.ipynb) - Non-adaptors fitting
  - [SEN_sigmoid_fixed_weight_scale.ipynb](./SEN_sigmoid_fixed_weight_scale.ipynb) - Sensitizers fitting
  - [Coin_toss_.ipynb](./Coin_toss_.ipynb) - Simulation of random inhibition connectivity to PCs
  - [Experimental data](./Experimental_data/Updated_again) - A folder with experimental data used to fit and test the model.

## System Requirements

### Hardware Requirements
This model requires only a standard computer with enough CPU performance. 

The runtime of one run is around 1.5 - 2 hours are generated using a computer with the next specs: (16 GB RAM, 13th Gen Intel(R) Core(TM) i7-1360P   2.20 GHz).
Runtime dependent on the number of function runs ("max_nfev" parameter) in each fitting algorythm.
### OS Requirements
The model designed and tested under the Windows operating systems. Test under Linux and Mac OS are needed to be done.

## Installation Guide

## Results

## Citation

For usage of the package and associated manuscript, please cite according to the enclosed (#TODO)


<b>Please note:</b>
<p>To work with this model download and unzip the main folder of this project onto your computer.</p>
<p><b>Important:</b> For all data uploades in the Jupyter notebook files to work correctly, please keep pathways and hierarchy of files and folders unchanged.</p>
<p>For working with a new data, provide specific file pathway to data files in data upload section. Data should be one-dimentional .txt file with each timepoint measurment in a new row. <b>Important:</b> timepoint of 0.164745 s wich corresponds to 6.07 Hz framerate recording, was used in our work. All data should be recorded in the same framerate, or timepoint parameter should be changed in all cells that begin with #timepoint comment.</p>

<p>Please find Phython setup requirements in the <i><b>requirements.txt</b></i> file. '3.11.5 | packaged by Anaconda, Inc. | (main, Sep 11 2023, 13:26:23) [MSC v.1916 64 bit (AMD64)]' Python version was used for development.</p>
<p></p>
<p><b>Files content:</b></p>
<ul>
  <li><i>Naive_opto_tails_prev_FBall_FBnotail_weightScalingTestTest.ipynb</i> - Naїve model fitting and optogenetic manipulation</li>
  <li><i>DEP_sigmoid_unfixed_with_heatmap_fixedSM.ipynb</i> - Depressors fitting</li>
  <li><i>NA_init.ipynb</i> - Non-adaptors fitting</li>
  <li><i>SEN_sigmoid_fixed_weight_scale.ipynb</i> - Sensitizers fitting</li>
  <li><i>Coin_toss_.ipynb</i> - Simulation of random inhibition connectivity to PCs</li>
</ul>
