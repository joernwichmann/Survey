This repository contains code that was used for the numerical experiments of the article:
	https://arxiv.org/abs/2412.14316

SETTING UP THE SIMULATION:
In order to run the simulations one first needs to install the finite element package
	FIREDRAKE	https://www.firedrakeproject.org/
After successful installation, start the virtual environment provided by firedrake:
	Navigate to the firedrake directory and use "source bin/activate" in the terminal. 
Next, install the package 	
	TQDM		https://tqdm.github.io/
This can be done by running "pip install tqdm"
The setup is complete and the virtual environment can be deactivated by running "deactivate"

SETTING UP DATA VISUALISATION:
Install the packages
	pandas		https://pypi.org/project/pandas/
	seaborn		https://pypi.org/project/seaborn/
This can be done by running "pip install pandas" and "pip install seaborn", respectively.

RUNNING THE SIMULATION:
We first need to start firedrake's virtual environment:
	Navigate to the firedrake directory and use "source bin/activate" in the terminal. 
Now, navigate to the directory 'gen-Stokes'.
The numerical simulations are started by running:
	python3 run_lid_driven_exp{1,2,3}.py
	python3 run_p_variation_exp{1,2,3}.py
After completion, we can deactivate the virtual environment.
	
RUNNING DATA VISUALISATION:
First, we need to process the data: 
	Navigate to 'data_processing/python-code-processing'.
	Select the data that should be processed by specifying the imported configs.
	Run in terminal: 
		python3 process_increment.py
After the processing, we visualise the data:
	Navigate to 'data_processing/python-code-plot'.
	Select the data that should be processed by specifying the imported configs.
	Run in terminal: 
		generate-{histogramm-plot,histo-plot-point-statistics,inc-plot,trajectory-plot,traj-point-statistics}.py
