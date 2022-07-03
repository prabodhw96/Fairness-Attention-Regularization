# Improving Fairness in Text Classification using Attention Regularization
## Overview
Natural language processing models exploit unintended correlations between gender and class labels thereby making the model biased. This project presents a new regularization-based method where attention layers are regularized to control their contribution to gender bias. It is built upon the causal mediation analysis by [Vig et al., 2020](https://arxiv.org/abs/2004.12265) and hence, follows a structural-behavioural analysis.
## Dependencies
Install the dependencies by running ``$ pip install -r requirements.txt``

To install PyTorch, go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and select the installation command for your environment.
## How to Run
``python main.py --dataset Twitter --batch_size 64 --num_epochs 15 --classifier_model bert-base-cased --seed 1 --analyze_results True --reg_coeff 0.7``
## Result
<img src="https://github.com/prabodhw96/Fairness-Attention-Regularization/raw/master/keywords.png" width="500" />