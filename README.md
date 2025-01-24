# CortexModel
A rate-based model of adaptation in layer 2/3 of mouse V1

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
