# Paper-PipeDefectSplitting
Code to reproduce results of paper: A Bayesian approach for CT reconstruction with defect detection of subsea pipelines.

# Main script
main.py is the main script where the compuations to reproduce the results in the paper are done and results saved.

The main script requires :
1) Installation of CUQIpy. This code runs on a pre-released version of CUQIpy. Install via command: pip install git+https://github.com/CUQI-DTU/CUQIpy.git@sprint16_add_JointModel 
2) Installation of Astra (https://github.com/astra-toolbox/astra-toolbox) 
3) Real or synthetic data, which is described below. 
4) The scripts funs.py and postprocessfuns.py included in this repository.

# Plotting
plotting.py creates the plots in the paper. 

The plotting script requires:
1) Saved results from numerical experiments in main.py
2) The script plotfuns.py included in this repository.

# Data
Synthetic data can be generated using the script generatesynthdata.py
Real data is found at https://doi.org/10.5281/zenodo.6817690.

# Total variation reconstruction
TV.py contains code to reproduce the TV reconstructions used for comparison in the paper. 
This script requires installation of CIL (https://tomographicimaging.github.io/CIL/nightly/index.html)
