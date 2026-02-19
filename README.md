# CortexModel

## Overview

A rate-based model of adaptation and behavioural state dependency in layer 2/3 of mouse V1 described in [Hinojosa, Kosiachkin *et al.* 2025](https://www.biorxiv.org/content/10.1101/2025.07.24.666602v2). This repository accompanies the manuscript and provides both a reference implementation of the model and the parameter optimization pipeline used in the study.

## Repo Contents
This repository is organized into two complementary components:
- The **model implementation**, used in the Jupyter notebooks at the root level, which illustrates how specific parameter choices determine neural responses and their dynamics.
- The **fitting pipeline**, located in the `Fitting/` directory, which was used to identify the parameter sets that reproduce experimental data accurately.

### Model implementation (Jupyter Notebooks)
  
  - [Model_V1_Rest.ipynb](./Model_V1_Rest.ipynb) - Model simulation using fitted connection weights to reproduce average traces on the dataset of resting mice.
  
  - [Model_V1_Loco.ipynb](./Model_V1_Loco.ipynb) - Model simulation using fitted connection weights to reproduce average traces on the dataset of locomoting mice.
    
  - [Model_V1_Rest_PCTypes_Dep.ipynb](./Model_V1_Rest_PCTypes_Dep.ipynb) - Model simulation for depressing PC subtype (Rest).
  - [Model_V1_Rest_PCTypes_Int.ipynb](./Model_V1_Rest_PCTypes_Int.ipynb) - Model simulation for intermediate PC subtype (Rest)
  - [Model_V1_Rest_PCTypes_Sen.ipynb](./Model_V1_Rest_PCTypes_Sen.ipynb) - Model simulation for sensitizing PC subtype (Rest)
    
  - [Model_V1_Loco_PCTypes_Dep.ipynb](./Model_V1_Loco_PCTypes_Dep.ipynb) - Model simulation for depressing PC subtype (Locomotion)
  - [Model_V1_Loco_PCTypes_Int.ipynb](./Model_V1_Loco_PCTypes_Int.ipynb) - Model simulation for intermediate PC subtype (Locomotion)
  - [Model_V1_Loco_PCTypes_Sen.ipynb](./Model_V1_Loco_PCTypes_Sen.ipynb) - Model simulation for sensitizing PC subtype (Locomotion)
    
  - [Experimental data](./Experimental_data) - Experimental data used for both model implementation and fitting.

The notebooks run deterministic simulations using previously fitted parameters. They do not perform optimization.

### Fitting Pipeline

  - [Fitting](./Fitting) - directory containing the optimization pipeline used to identify parameter sets that reproduce the experimental data with high-performance computing (Artemis).

## System Requirements

### Hardware Requirements
**Model implementation** requires only a standard computer with enough CPU performance. 

The runtime of a single simulation is few seconds on a standard desktop computer (16 GB RAM, 13th Gen Intel(R) Core(TM) i7-1360P 2.20 GHz).

The **fitting pipeline** was executed on the Artemis high-performance computing cluster. Runtime depends on the number of function evaluations (max_nfev), the number of initial conditions used in the optimization, and the number of CPU cores allocated. See the Fitting/ README for details.

### OS Requirements

The model was designed and tested under the Windows operating system.

## Installation Guide
The model was developed using Python 3.11.5 | packaged by Anaconda, Inc. 
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

To install the **model implementation** and work with the Jupyter Notebook files download and unzip the main folder of this GitHub repository.
<p><b>Important:</b> For all data uploads in the Jupyter notebook files to work correctly, please keep pathways and hierarchy of files and folders unchanged.</p>
<p>For working with different datasets, provide specific file pathway to data files in data upload section. Data should be a one-dimensional .txt file with one data point per row. **Important:** a frame interval of 0.164745 s (correspoding to a 6.07 Hz acquisition rate) was used in our work. Please adjust the frame interval according to your recording framerate.</p>

For instructions on how to run the **fitting pipeline** using Artemis, check the Readme file in the /Fitting directory.

## Results
In [Model_V1_Rest.ipynb](./Model_V1_Rest.ipynb) and [Model_V1_Loco.ipynb](./Model_V1_Loco.ipynb) we fit the data to average traces of four main populations in layer 2/3 of mouse V1 (Figure 6) and fine-tuned with optogenetic effects (Figure 5) for two behavioural states: rest and locomotion.

Next in [Model_V1_Rest_PCTypes_Dep.ipynb](./Model_V1_Rest_PCTypes_Dep.ipynb), [Model_V1_Rest_PCTypes_Int.ipynb](./Model_V1_Rest_PCTypes_Int.ipynb), and [Model_V1_Rest_PCTypes_Sen.ipynb](./Model_V1_Rest_PCTypes_Sen.ipynb) we fit depressing, intermediate, and sensitizing PC cell subpopulations for resting state and similarly in [Model_V1_Loco_PCTypes_Dep.ipynb](./Model_V1_Loco_PCTypes_Dep.ipynb), [Model_V1_Loco_PCTypes_Int.ipynb](./Model_V1_Loco_PCTypes_Int.ipynb), and [Model_V1_Loco_PCTypes_Sen.ipynb](./Model_V1_Loco_PCTypes_Sen.ipynb) - for locomoting state (Figure 7).

## Citation

For usage of the model and associated manuscript, please cite according to the enclosed [Hinojosa, Kosiachkin *et al.* 2025](https://www.biorxiv.org/content/10.1101/2025.07.24.666602v2)


