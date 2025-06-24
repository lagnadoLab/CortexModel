# CortexModel(Updating currently)

# Readme and Subpopulatuion part is currently under update due to new paper submission. Should be finished by the end of 25/06/2025

## Overview
A rate-based model of adaptation in layer 2/3 of mouse V1 described in [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.12.13.628375v1) manuscript.

## Repo Contents

  - [Naive_opto_tails_prev_FBall_FBnotail_weightScalingTestTest.ipynb](./Naive_opto_tails_prev_FBall_FBnotail_weightScalingTestTest.ipynb) - Naїve model fitting average traces and performing optogenetic manipulation
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
The model is developed using Python 3.11.5 | packaged by Anaconda, Inc. 
The full virtual environment requirements are in [requirements.txt](./requirements.txt).
The main dependencies that should be installed/updated are:
```
ipykernel==6.28.0
ipython==8.25.0
ipywidgets==8.1.3
lmfit==1.3.2
matplotlib==3.9.1.post1
matplotlib-inline==0.1.6
numdifftools==0.9.41
numpy==2.0.1
pandas==2.2.2
scikit-learn==1.5.1
scipy==1.14.0
seaborn==0.13.2
```

To install the model and work with the Jupyter Notebook files download and unzip the main folder of this GitHub folder.
<p><b>Important:</b> For all data uploades in the Jupyter notebook files to work correctly, please keep pathways and hierarchy of files and folders unchanged.</p>
<p>For working with a new data, provide specific file pathway to data files in data upload section. Data should be one-dimentional .txt file with each timepoint measurment in a new row. <b>Important:</b> timepoint of 0.164745 s wich corresponds to 6.07 Hz framerate recording, was used in our work. All data should be recorded in the same framerate, or timepoint parameter should be changed in all cells that begin with #timepoint comment.</p>

## Results
In [Naive_opto_tails_prev_FBall_FBnotail_weightScalingTestTest.ipynb](./Naive_opto_tails_prev_FBall_FBnotail_weightScalingTestTest.ipynb) we fit the data to average traces of four main populations in layer 2/3 of mouse V1 (Figure 5) and tested optogenetic effects (Figure 6).
Next in [DEP_sigmoid_unfixed_with_heatmap_fixedSM.ipynb](./DEP_sigmoid_unfixed_with_heatmap_fixedSM.ipynb), [NA_init.ipynb](./NA_init.ipynb), and [SEN_sigmoid_fixed_weight_scale.ipynb](./SEN_sigmoid_fixed_weight_scale.ipynb) we fit depressing, non-adapting, and sensitizing PC cell subpopulations, respectively (Figure 7).
Finally, in [Coin_toss_.ipynb](./Coin_toss_.ipynb) we perform simulation of random inhibition connectivity to PC cells in layer 2/3 of mouse V1, based on literature, and comparing to results obtained from the model (Figure 8).
## Citation

For usage of the model and associated manuscript, please cite according to the enclosed [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.12.13.628375v1)


