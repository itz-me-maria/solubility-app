######################
# Import libraries
######################
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors

######################
# Custom function
######################
## Calculate molecular descriptors
def AromaticProportion(m):
  aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
  aa_count = []
  for i in aromatic_atoms:
    if i==True:
      aa_count.append(1)
  AromaticAtom = sum(aa_count)
  HeavyAtom = Descriptors.HeavyAtomCount(m)
  #AR = AromaticAtom/HeavyAtom
  AR = AromaticAtom/HeavyAtom if HeavyAtom > 0 else 0
  return AR

def generate(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData= np.arange(1,1)
    i=0
    for mol in moldata:

        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
        desc_AromaticProportion = AromaticProportion(mol)

        row = np.array([desc_MolLogP,
                        desc_MolWt,
                        desc_NumRotatableBonds,
                        desc_AromaticProportion])

        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1

    columnNames=["MolLogP","MolWt","NumRotatableBonds","AromaticProportion"]
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)

    return descriptors

######################
# Page Title
######################

image = Image.open('solubility-logo.png')

st.image(image, use_container_width=True)


st.markdown("""
**Developed by:** Maria Roychan

**Project Goal:** To predict the solubility of drug-like molecules using machine learning.

This app predicts the **Solubility (LogS)** values of molecules based on their molecular descriptors.

""")

######################
# Input molecules (Side Panel)
######################

if "smiles_text" not in st.session_state:
    st.session_state["smiles_text"] = ""
if "clear_count" not in st.session_state:
    st.session_state["clear_count"] = 0

st.sidebar.header('User Input Features')

SMILES = st.sidebar.text_area("SMILES INPUT",
    placeholder="Enter SMILES here",
    key=f"main_smiles_{st.session_state['clear_count']}")

col1, col2 = st.sidebar.columns(2)
with col1:
    search = col1.button("Search")
with col2:
    if col2.button("Clear"):
        st.session_state["clear_count"] += 1
        st.rerun()


st.session_state["smiles_text"] = SMILES
if not search or not SMILES:
    st.stop()

SMILES = "C\n" + SMILES #Adds C as a dummy, first item
SMILES = SMILES.split('\n')

st.header('Input SMILES')
SMILES[1:] # Skips the dummy first item


## Calculate molecular descriptors
st.header('Computed molecular descriptors')
SMILES = [s for s in SMILES if s.strip() != ""]

invalid = [s for s in SMILES if Chem.MolFromSmiles(s) is None]
SMILES = [s for s in SMILES if Chem.MolFromSmiles(s) is not None]

if invalid:
    st.error("⚠️ Please enter SMILES in the correct form!")
    st.stop()

if not SMILES:
    st.error("⚠️ Please enter SMILES in the correct form!")
    st.stop()
X = generate(SMILES)
X[1:] # Skips the dummy first item





######################
# Pre-built model
######################

# Reads in saved model
load_model = pickle.load(open('solubility_model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_model.predict(X)
#prediction_proba = load_model.predict_proba(X)

st.header('Predicted LogS values')
prediction[1:] # Skips the dummy first item
