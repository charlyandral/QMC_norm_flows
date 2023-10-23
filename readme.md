# Project Name

Code the paper "Combining Normalizing Flows and Quasi-Monte Carlo"

## Installation

To run, the code needs two slightly modified versions of the packages fab-torch (https://github.com/lollcat/fab-torch) and FlowMC (https://github.com/kazewong/flowMC) to use QMC sampling. The modified versions are included in the folder `fab-torch` and `flowMC` respectively. To install, run a `pip install -e .` in each folder. A direct install of the unmodified versions from github will lead to errors. 

The code used to generate the figures are located in two folders, section 5.1 and section 5.2



## Usage


The code used to generate the figures are located in two folders, section 5.1 and section 5.2

### Section 5.1
All the code is in a single notebook, fab_gmm.ipynb

### Section 5.2.2
run_dim.py generates the dataframe with the raw date, analysis_dim.py runs the analysis from the dataframe

### Section 5.2.3
run_archi.py generates the dataframe with the raw date, analysis_archi.py runs the analysis from the dataframe

### Section 5.2.4
run_sequence.py generates the dataframe with the raw date, analysis_sequence.py runs the analysis from the dataframe

### Section 5.2.5
For this section, the dataframe used is the one generated in section 5.2.4. The code for the analysis is in analysis_MCMC.py



