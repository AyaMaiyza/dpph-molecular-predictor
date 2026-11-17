import numpy as np
from rdkit import Chem,DataStructs,rdBase
from rdkit.Chem import Descriptors,rdFingerprintGenerator,Lipinski,rdMolDescriptors
NAMES=['MolWt','MolLogP','TPSA','HBD','HBA','RotatableBonds','RingCount','FractionCSP3','HeavyAtomCount','CarbonCount','OxygenCount','AromaticAtomCount']
def encode(smiles):
 gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
 bits=[];counts=[];desc=[]
 for smi in smiles:
  m=Chem.MolFromSmiles(smi)
  if m is None:raise ValueError('Invalid SMILES')
  b=np.zeros(2048,dtype=np.uint8);c=np.zeros(2048,dtype=np.uint32)
  DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m),b)
  DataStructs.ConvertToNumpyArray(gen.GetCountFingerprint(m),c)
  d=[Descriptors.MolWt(m),Descriptors.MolLogP(m),rdMolDescriptors.CalcTPSA(m),Lipinski.NumHDonors(m),Lipinski.NumHAcceptors(m),Lipinski.NumRotatableBonds(m),rdMolDescriptors.CalcNumRings(m),rdMolDescriptors.CalcFractionCSP3(m),m.GetNumHeavyAtoms(),sum(a.GetAtomicNum()==6 for a in m.GetAtoms()),sum(a.GetAtomicNum()==8 for a in m.GetAtoms()),sum(a.GetIsAromatic() for a in m.GetAtoms())]
  bits.append(b);counts.append(c);desc.append(d)
 return np.asarray(bits),np.asarray(counts),np.asarray(desc,dtype=float)
