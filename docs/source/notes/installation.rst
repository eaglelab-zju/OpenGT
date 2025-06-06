Installation
=============
Python environment setup with Conda
---------------------------------------

.. code-block:: bash

	conda create -n opengt python=3.10
	conda activate opengt

	pip install torch==2.5 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

	pip install torch_geometric

	pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

	# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
	conda install openbabel fsspec rdkit

	pip install pytorch-lightning yacs torchmetrics
	pip install performer-pytorch
	pip install tensorboardX
	pip install ogb
	pip install wandb
	pip install pymetis
	pip install opt-einsum

	conda clean --all
