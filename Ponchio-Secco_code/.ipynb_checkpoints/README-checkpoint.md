In "Architectures_code" we put the necessary code to run the various models taken from the respective GitHub repositories.
The code is mostly the same as the original one, with minor modifications for compatibility.
Each model has its own directory.

In "Our_code" we put the code wrote to run the various models and extract meaningful metrics, in particular:

	- "Partial_Convolution.ipynb": Partial Convolution architecture with pre-trained weights.
			reference paper: https://arxiv.org/abs/1804.07723
			reference repository: https://github.com/naoto0804/pytorch-inpainting-with-partial-conv
			
	- "Contextual_Attention.ipynb": Contextual Attention architecture with pre-trained weights.
			reference paper: https://arxiv.org/abs/1801.07892
			reference repository: https://github.com/daa233/generative-inpainting-pytorch

	- "run_Iizuka.py": Iizuka model script for our data format.
			reference paper: https://api.semanticscholar.org/CorpusID:3707204

	- "Metrics.ipynb": Our defined metrics for measuring the performance of the networks

	- "Metrics_analysis_comparison_all_models.ipynb": Results of the metrics over all models put together.