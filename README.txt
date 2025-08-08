This repository contains the codes that were used for the numerical experiments of the article:
	https://arxiv.org/abs/2508.05564

--------------------------------------------------------------------------------------------
SETTING UP THE SIMULATION:
In order to run the simulations one first needs to install the finite element package
	FIREDRAKE (Version:2025.4.1)		https://www.firedrakeproject.org/
After successful installation, start the virtual environment provided by firedrake:
	Navigate to the firedrake directory and use ". venv-firedrake/bin/activate" in the terminal. 
Next, install the package 	
	TQDM (Version: 4.67.1)			https://tqdm.github.io/
This can be done by running "pip install tqdm"
The setup is complete and the virtual environment can be deactivated by running "deactivate"

RUNNING THE SIMULATION:
We first need to start firedrake's virtual environment:
	Navigate to the firedrake directory and use ". venv-firedrake/bin/activate" in the terminal. 
Now, navigate to the directory 'Survey/experiments'. This folder contains the codes not only for running the experiments of the article but many other experiments too.
To repeat one of the numerical simulations, navigate to the end of the directory structure and execute "python3 run_{'to be filled appropriately'}". 
	
--------------------------------------------------------------------------------------------
The visualisation of the 'lid-driven cavity' experiment uses the following:

SETTING UP DATA VISUALISATION:
Install the packages
	pandas (Version: 2.1.4+dfsg)		https://pypi.org/project/pandas/
	seaborn	(Version: 0.13.2)		https://pypi.org/project/seaborn/
This can be done by running "pip install pandas" and "pip install seaborn", respectively.
	
RUNNING DATA VISUALISATION:
	Navigate to 'Survey/experiments/NSE_lid_driven_cavity/Crank_Nicolson/data_processing/python-code-plot'.
	Select the data that should be processed by specifying the imported configs.
	Run in terminal: 
		generate-{histogramm-plot,histo-plot-point-statistics,trajectory-plot,traj-point-statistics}.py
