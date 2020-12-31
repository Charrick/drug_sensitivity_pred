# drug_sensitivity_pred

A Deep Learning Model for Prediction of sensitivity of drug in cellline.

This repository contains script which were used to build and train the deep model.

# Scripts
main_model.py - This script is used to build and train the model and evaluate the performance of 10_fold_cv of the model.

MaskedLinear.py - This script is used to control the connection of the pathway layer

cacul_r.py - This script is used to calculate the Pearson correlation coefficient

# Data

The ‘data’ folder contains the data files that the scripts depends on.

nor_feature.csv - Normalized feature dataset.

relation.csv - Connection relationship table between genes and pathways

# Citations

If you use this project for your research, or incorporate our learning algorithms in your work, please cite:

Deng, L. , Cai, Y. , Zhang, W. , Yang, W. , & Liu, H. . (2020). Pathway-guided deep neural network toward interpretable and predictive modeling of drug sensitivity. Journal of Chemical Information and Modeling, XXXX(XXX).
