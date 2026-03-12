# Model Fitting and Optogenetic Validation Pipeline

## Overview
This pipeline reproduces the parameter fitting and optogenetic validation described in [Hinojosa, Kosiachkin *et al.* 2025](https://www.biorxiv.org/content/10.1101/2025.07.24.666602v2). Parameter sweeps were executed in parallel on a SLURM-based high-performance computing (HPC) cluster.

To install the **model fitting** and work with the python files download and unzip all the folders including Fitting and Experimental data. Copy these folders in your HPC server to run fittings using bash language and python.
<p><b>Important:</b> For all scripts to work correctly, please keep pathways, hierarchy of files and folders unchanged.</p>

The workflow consists of:

  1. **Global sampling**
        1. Sobol sampling of initial parameter space.
        2. Cluster-based parameter sweep
        3. Performance-based filtering

  2. **Local refinement**

  3. **Optogenetic validation**

  4. **Adaptation direction**

  5. **Post-hoc clustering analysis of good solutions**

## Results and figures

First the model was fitted to the activity traces of 4 neuronal populations in V1 during rest and locomotion: pyramidal cells (PCs), somatostatin interneurons (SSTs), parvalbumin interneurons (PVs) and VIP interneurons (Figure 5). These results were tested using optogenetics (Figure 4). Finally, the connectivity of different PC adaptive types was assessed with the model (Figure 6).

## Environment Setup

Create the python environment for the first time, installing the packages in requirements.txt:

python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt

The main dependencies that should be installed/updated and the version we used are:
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
## 1. Global sampling

Initial conditions homogenously distributed across parameter space were used to identify sets of solutions that fit the experimental data (good solutions).

## 1.1. Parameter Sampling (Sobol)

**Description**: Generates quasi-random Sobol initial parameter sampling homogenously across the parameter space to fit neuronal responses.

**Script**: Sobol_test.py

**Output**: sobol_samples.csv

## 1.2. Cluster Execution (SLURM)

**Description:** This step runs the core model fitting. The equations are fitted to neuronal responses starting from the initial conditions provided in `sobol_sample.csv`. Each parameter set is evaluated in parallel by distributing jobs across multiple cores using SLURM.

**Submit**: sbatch **submit_ID_sweep.sh**. This will initiate **model_run.py** which will use **model_functions.py**.

**Outputs**: Per-solution CSV result files and Log files (terminal output and errors)

These scripts were executed on the University of Sussex Artemis HPC cluster using SLURM, but can be adapted to any SLURM-based system.

## 1.3. Combine Results and filter good fits

**Description**: combine parameters from all solutions into one only csv file and select those solutions with chi-square below threshold. Rest threshold = 3, Locomotion threshold = 10.

**Script**: combine_results.py --condition rest or locomotion

**Output**: good_fits_condition.csv

## 2. Local refinement

**Description**: The same three-step procedure performed in Global sampling (Sobol sampling, cluster-based sweep, and performance filtering) was repeated within a restricted parameter region defined as ±2 SD around the globally found good solutions.

**Scripts**: Sobol_test_local.py, submit_ID_sweep.sh, combine_result.py.

**Output**: good_fits_condition.csv

## 3. Optogenetic Validation

**Description**: test of good solutions with optogenetic modulation of inhibitory interneurons. The model was required to fit the activity of PCs when activating and silencing PVs and SST interneurons(Figure 4).

**Scripts**: sbatch **submit_opto.sh**. This will initiate **Opto_run_loco.py** for each solution. Finally, combine_results_opto.py will combine all fits and filter the good ones.

**Output**: good_fits_opto.csv

## 5. Adaptation direction
**Description**: the model was constrained to fit the activity of PC supopulations with different adaptive properties: sensitizers, intermediates and depressors(Figure 6).

**Scripts**: sbatch **submit_ID_DepSen.sh** that will initiate **Model_run_DepSen.py** for each solution. Finally, combine_results_DepSen.py will combine all fits and filter the good ones.

**Output**: good_fits_sensitizers.csv, good_fits_intermediates.csv, good_fits_depressors.csv


## 5. Post-hoc Parameter Structure Analysis

**Description**: Good solutions were averaged and their structure in parameter space was analyzed using clustering methods.

**Scripts**:ExtractWeightsAvg.py

**Output**:ConnWeights.csv