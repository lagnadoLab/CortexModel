# CortexModel

## Overview

A rate-based model of adaptation and behavioural state dependency in layer 2/3 of mouse V1 described in <b><i>[reference placeholder]</i></b> manuscript.

## Repo Contents

  - [S1_hab_loco_7_2.ipynb](./S1_hab_loco_7_2.ipynb) - The model fitting average traces and performing optogenetic manipulation on the dataset of locomoting mice.
  - [S1_Hab_NoLoco_9_1.ipynb](./S1_Hab_NoLoco_9_1.ipynb) - The model fitting average traces and performing optogenetic manipulation on the dataset of resting mice.
    
  - [C_S1_hab_Dep_NoLoco_1_paper.ipynb](./C_S1_hab_Dep_NoLoco_1_paper.ipynb) - Depressors fitting (Rest)
  - [C_S1_hab_NA_NoLoco_1_paper.ipynb](./C_S1_hab_NA_NoLoco_1_paper.ipynb) - Non-adaptors fitting (Rest)
  - [C_S1_hab_SEN_NoLoco_1_paper.ipynb](./C_S1_hab_SEN_NoLoco_1_paper.ipynb) - Sensitizers fitting (Rest)
    
  - [C_S1_hab_Dep_loco_1_paper.ipynb](./C_S1_hab_Dep_loco_1_paper.ipynb) - Depressors fitting (Locomotion)
  - [C_S1_hab_NA_loco_1_WeightTest_paper.ipynb](./C_S1_hab_NA_loco_1_WeightTest_paper.ipynb) - Non-adaptors fitting (Locomotion)
  - [C_S1_hab_SEN_loco_1_2_paper.ipynb](./C_S1_hab_SEN_loco_1_2_paper.ipynb) - Sensitizers fitting (Locomotion)
    
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
<p>For working with a new data, provide specific file pathway to data files in data upload section. Data should be one-dimentional .txt file with each timepoint measurment in a new row. <b>Important:</b> timepoint of 0.164745 s wich corresponds to 6.07 Hz framerate recording, was used in our work. All data should be recorded in the same framerate, or timepoint parameter, where it is iniciated should be changed.</p>

## Results
In [S1_hab_loco_7_2.ipynb](./S1_hab_loco_7_2.ipynb) and [S1_Hab_NoLoco_9_1.ipynb](./S1_Hab_NoLoco_9_1.ipynb) we fit the data to average traces of four main populations in layer 2/3 of mouse V1 (Figure 6) and fine tuned with optogenetic effects (Figure 5) for two behavioural states: locomotion and rest.
Next in [C_S1_hab_Dep_NoLoco_1_paper.ipynb](./C_S1_hab_Dep_NoLoco_1_paper.ipynb), [C_S1_hab_NA_NoLoco_1_paper.ipynb](./C_S1_hab_NA_NoLoco_1_paper.ipynb), and [C_S1_hab_SEN_NoLoco_1_paper.ipynb](./C_S1_hab_SEN_NoLoco_1_paper.ipynb) we fit depressing, non-adapting, and sensitizing PC cell subpopulations for resting state and similarly in [C_S1_hab_Dep_loco_1_paper.ipynb](./C_S1_hab_Dep_loco_1_paper.ipynb), [C_S1_hab_NA_loco_1_WeightTest_paper.ipynb](./C_S1_hab_NA_loco_1_WeightTest_paper.ipynb), and [C_S1_hab_SEN_loco_1_2_paper.ipynb](./C_S1_hab_SEN_loco_1_2_paper.ipynb) - for locomoting state (Figure 7).

## Citation

For usage of the model and associated manuscript, please cite according to the enclosed <b><i>[reference placeholder]</i></b>


