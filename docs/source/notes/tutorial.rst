Tutorial
=============
Running Experiments
---------------------
1. Activate Environment

First, activate the project environment:

.. code-block:: bash

	conda activate opengt



2. Run a Single Experiment

Use `main.py` along with a configuration file to launch an experiment. Configuration files are stored in: ``configs/<ModelName>/<DatasetName-ModelName>.yaml``

Example: Run DIFFormer on Cora (3 runs)

.. code-block:: bash

	python main.py --cfg configs/DIFFormer/cora-DIFFormer.yaml --repeat 3


Results will be saved automatically to: ``results/DIFFormer/cora-DIFFormer/``


3. Hyperparameter Tuning

To perform grid search on hyperparameters, use the provided `run_batch.sh` script and corresponding grid files under the `grids/` directory.


.. code-block:: bash

	bash run_batch.sh


Please note that the grid file and the configuration file names should be modified in the script.

Each line of the grid file should be written in the following format:

.. code-block::

	<Config parameter name> <Display name> <List of possible values>


For example:

.. code-block::

	gt.layers nlayer [1,2,3,4]
	gt.aggregate agg ['add', 'cat']
	gt.dropout dropout [0.2, 0.5, 0.8]


This will explore all parameter combinations defined in the relevant grid configuration.


4. Aggregating Results

To summarize and aggregate results from multiple runs:

.. code-block:: bash

	python agg_test.py


This script collects results across seeds and outputs averaged performance metrics with standard deviations in to a `.csv` file.