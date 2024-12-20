import argparse
import pandas as pd

# Create the parser
parser = argparse.ArgumentParser(description='Process some CSV files.')

# Add the arguments
parser.add_argument('input_file', type=str, help='The input CSV file')

# Parse the arguments
args = parser.parse_args()

# Read the input file into a DataFrame
df = pd.read_csv(args.input_file)

class MuLoss(tf.keras.losses.Loss):
    def __init__(self, name="mu_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        pmu, pnA, pnB, pnC, pnD, T = tf.split(y_pred, [1, 1, 1, 1, 1, 1], axis=1)
        mu, nA, nB, nC, nD = tf.split(y_true, [1, 1, 1, 1, 1], axis=1)
        
        pA = pnA * (-1.9879 + 20.7554) - 20.7554
        pB = pnB * (4240.2 - 415.84) + 415.84
        pC = pnC * (0.037091 - 0.000474) + 0.000474
        pD = pnD * (-4.01E-8 + 3.03E-5) - 3.03E-5
        
        y_pred_mu = pA + pB / T + pC * T + pD * T**2

        loss_A = tf.square(nA - pnA)
        loss_B = tf.square(nB - pnB)
        loss_C = tf.square(nC - pnC)
        loss_D = tf.square(nD - pnD)
        loss_mu = tf.square(mu - pmu)
        loss_pred_mu = tf.square(pmu - y_pred_mu)

        total_loss = loss_A + loss_B + loss_C + loss_D + loss_mu + 0.1*(loss_pred_mu)

        return total_loss

from tensorflow.keras.optimizers import Adam

n_1 = 512
n_2 = 512
n_3 = 128
n_4 = 64

# Define the model architecture
A1 = Input(shape=(1944,), name='A1')
A2 = Dense(n_1, activation='relu', name='A2')(A1)
A3 = Dense(n_2, activation='relu', name='A3')(A2)
A4 = Dense(n_3, activation='relu', name='A4')(A3)
A5 = Dense(n_4, activation='relu', name='A5')(A4)
A6 = Dense(5, name='A6')(A5)

T = Input(shape=(1,), name='T')
combined = Concatenate()([A6, T])

model = Model(inputs=[A1, T], outputs=combined)

# Compile the model with the custom loss function
model.compile(loss=MuLoss(), optimizer=Adam())

# Set the seed for reproducibility
seed(1)
tf.random.set_seed(1*7+333)


import tensorflow as tf
model.load_weights('PINN.h5')

import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import DataStructs

import seaborn as sns
from sklearn.model_selection import train_test_split


from rdkit.Chem import rdMolDescriptors as rdmd
from tqdm import tqdm
from functools import wraps

Corr_df = pickle.load(open("Corr_Visc.pickle","rb"))
unique_list = pickle.load(open("unique_list_Visc.pickle","rb"))
Columns = pickle.load(open("Columns_Visc.pickle","rb"))
Substructure_list = pickle.load(open("polymer.keys_Visc.pickle","rb"))

molecules = df.Smiles.apply(Chem.MolFromSmiles)
fp = molecules.apply(lambda m: AllChem.GetMorganFingerprint(m, radius=3))
fp_n = fp.apply(lambda m: m.GetNonzeroElements())
MY_finger = []
for polymer in fp_n:
    my_finger = [0] * len(unique_list)
    for key in polymer.keys():
        if key in list(Corr_df[0]):
            index = Corr_df[Corr_df[0] == key]['index'].values[0]
            my_finger[index] = polymer[key]         
    MY_finger.append(my_finger)
X_MD = pd.DataFrame(MY_finger)
X_MD = X_MD[Columns]
T_values = df["T (K)"].values.astype(float)
X_MD = np.column_stack((X_MD, T_values))
X_MD = np.column_stack((X_MD, T_values))

n_samples, n_features = X_MD.shape

slice_index = n_features - 1

X_MD_1 = X_MD.iloc[:, :slice_index] 
X_MD_2 = X_MD.iloc[:, slice_index:] 

score = model.predict(([X_MD_1, X_MD_2]))

df['log(Visc)'] = score

df.to_csv('Prediction_results.csv',index=False)