# Alternate-Reward Reinforcement Learning Guided Drug Molecule Generation for High Binding Affinity to Novel Target
Design of new inhibitors for novel targets is a very important problem especially in the current scenario. Conventional approaches undertaken to this end like high-throughput virtual screening require extensive combing through existing datasets in the hope of finding possible matches. In this study we propose a computational strategy for de novo generation of molecules with high binding affinities to the specified target. A deep generative model is built using a stack augmented recurrent neural network for initially generating drug like molecules and then it is optimized using reinforcement learning to start generating molecules with desirable properties the most important of which is the binding affinity. The reinforcement learning section of the pipeline is further extended to multi-objective optimization showcasing the model's ability to generate molecules with a wide variety of properties desirable for drug like molecules, like, LogP, Quantitative Estimate of Drug Likeliness etc. For multi-objective optimization, we have devised a novel strategy for optimization in which the property being used to calculate the reward is changed periodically. In comparison to the conventional approach of taking a weighted sum of all rewards, this strategy has shown much better performance in it's ability to generate a significantly higher number of molecules with desirable properties.

![Pipeline](Images/MainDiagram.png "Pipeline")

This repository contains the code for optimization of the generator model using predictor machine learning models and docking calculations as well as other properties.

Installation
------------
[Install miniconda](https://docs.conda.io/en/latest/miniconda.html) and run the following command.

```conda env create --file environment.yml```

Running Experiments
--------------
Open the `Optimizer` directory.

To run each of the experiments run the following commands
- Single Objective: Binding Affinity with SARS-CoV-2 $M_{pro}$ using Docking Calculation
```python model_logP_QED_switch.py --reward_function exponential --num_iterations 175 --use_wandb yes --predictor dock --protein 6LU7 --remarks <remarks>```
- Single Objective: Binding Affinity with TTBK1 using Docking Calculation
```python model_logP_QED_switch.py --reward_function exponential --num_iterations 175 --use_wandb yes --predictor dock --protein 4BTK --remarks <remarks>```
- Single Objective: Binding Affinity with TTBK1 using GIN
```python model_logP_QED_switch.py --reward_function exponential --num_iterations 175 --use_wandb yes --predictor gin --protein 4BTK --remarks <remarks>```
- Multi Objective : Binding Affinity with SARS-CoV-2 $M_{pro}$ using Docking Calculation and target LogP = 2.5 (sum)
```python model_logP_QED_switch.py --reward_function exponential --num_iterations 175 --use_wandb yes --predictor dock --protein 6LU7 --remarks <remarks> --logP yes --logP_threshold 2.5 --switch no```
- Multi Objective : Binding Affinity with TTBK1 using Docking Calculation and target LogP = 2.5 (sum)
```python model_logP_QED_switch.py --reward_function exponential --num_iterations 175 --use_wandb yes --predictor dock --protein 4BTK --remarks <remarks> --logP yes --logP_threshold 2.5 --switch no```
- Multi Objective : Binding Affinity with TTBK1 using GIN and target LogP = 2.5 (sum)
```python model_logP_QED_switch.py --reward_function exponential --num_iterations 175 --use_wandb yes --predictor gin --protein 4BTK --remarks <remarks> --logP yes --logP_threshold 2.5 --switch no```
- Multi Objective : Binding Affinity with TTBK1 using GIN, target LogP = 2.5, target QED = 1, target TPSA = 100 $\AA^{2}$ and target $\Delta G_{Hyd}$ = -10 kcal/mol (sum)
```python model_logP_QED_switch.py --reward_function exponential --num_iterations 175 --use_wandb yes --predictor gin --protein 4BTK --remarks <remarks> --logP yes --logP_threshold 2.5 --qed yes --qed_threshold 1 --solvation yes --solvation_threshold -10 --tpsa yes --tpsa_threshold 100 --switch no```
