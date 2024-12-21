# FNN-and-PINN-for-Viscosity-Prediction
Machine learning models for paper "Assessing the Effectiveness of Neural Networks and Molecular Dynamics Simulations in Predicting Viscosity: Insights for Tailored Molecular Design"

The pre-trained models needed for prediction can be found at: https://zenodo.org/records/14538211

To predict polymer viscosity using FNN and PINN models, save the dataframe containing the polymer's SMILES information and temperature into a .csv file, and then use a command like:

`python FNN.py xxx.csv`

`python PINN.py xxx.csv`
