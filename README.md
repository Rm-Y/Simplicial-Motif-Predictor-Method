# Simplicial-Motif-Predictor-Method
This repository contains the implementation of the Simplicial Motif Predictor Method (SMPM), proposed in the paper Simplicial Motif Predictor Method for Higher-Order Link Prediction.
## Overview

In this paper, we employ motifs for higher-order link prediction in simplicial networks. The main contributions include:

	1. Introducing simplicial motifs, which can distinguish higher-order multi-interactions and capture rich local information.
 
	2. Proposing a supervised learning model, SMPM, that uses simplicial motifs as predictors for open and closed simplices.
 
	3. Implementing, training, and evaluating SMPM across temporal simplicial networks to validate its effectiveness in predicting higher-order links.

## Dataset
The dataset used in the experiments is from the paper:
```
Benson, Austin R., et al. “Simplicial closure and higher-order link prediction.” Proceedings of the National Academy of Sciences 115.48 (2018): E11221-E11230.
```

## Contents
This repository includes:

1. **`read_simplices_data.py`**  
   The code processes simplicial network data,  and categorizes simplices into open and closed triangles.

2. **`compute_motifs.py`**  
   The code calculates 25 simplicial motifs (M1 to M16) based on the edge list and neighbor relationships in a simplicial network.

3. **`construct_motif_feature.py`**  
   The code computes motif-based features for a graph, including simple, geometric, and harmonic means, and saves the processed feature sets for training and testing datasets.

4. **`train_model.py`**  
   The code trains and evaluates model (such as Logistic Regression) on multiple datasets, performs data balancing through undersampling, and computes evaluation metrics.
   
## Citation

If you use this algorithm in your research, please cite this project.
```
Yang, Rongmei, Bo Liu, and Linyuan Lü. "Simplicial motif predictor method for higher-order link prediction." Expert Systems with Applications (2024): 126284.
```

