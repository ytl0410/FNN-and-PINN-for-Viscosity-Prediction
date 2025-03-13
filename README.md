# FNN-and-PINN-for-Viscosity-Prediction
Machine learning models for paper "Assessing the Effectiveness of Neural Networks and Molecular Dynamics Simulations in Predicting Viscosity: Insights for Tailored Molecular Design"

The pre-trained models needed for prediction can be found at: https://zenodo.org/records/14538211

To use the pre-trained models, before running, please set up your development environment: ensure that all necessary libraries and dependencies are installed. You can then run:
```
conda create -n py39 python=3.9
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python -m pip install "tensorflow<2.11"
pip install numpy==1.26.4
pip install matplotlib==3.8.0
pip install pandas==1.5.3
pip install rdkit==2023.9.1
pip install scikit-learn==1.3.0
pip install keras-tuner==1.4.5
pip install seaborn==0.13.0
```

To predict polymer viscosity using FNN and PINN models, save the dataframe containing the polymer's SMILES information and temperature into a .csv file, and then use a command like:

`python FNN.py xxx.csv`

`python PINN.py xxx.csv`
