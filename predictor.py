from pathlib import Path
import csv,io,json,hashlib,sys,threading
import numpy as np,joblib,sklearn
from rdkit import Chem,rdBase
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from features import encode
HERE=Path(__file__).resolve().parent
MODEL_NAMES=['ET_count_fixed','SVM_count_inner_C']

def require(ok,message):
 if not ok:raise ValueError(message)

def read_csv(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def sha(data):return hashlib.sha256(data).hexdigest()

def validate_smiles(value):
 value=str(value).strip();require(value and len(value)<=5000,'Enter a SMILES string of1–5000 characters.')
 mol=Chem.MolFromSmiles(value);require(mol is not None,'Invalid SMILES.')
 require(len(Chem.GetMolFrags(mol))==1,'Enter one connected compound; mixtures and disconnected salts are not supported.')
 require(1<=mol.GetNumHeavyAtoms()<=300,'Input limit:1–300 heavy atoms. This limit is computational, not an applicability cutoff.')
 return Chem.MolToSmiles(mol,isomericSmiles=True),Chem.MolToSmiles(mol,isomericSmiles=False)

def parse_queries(text):
 rows=list(csv.DictReader(io.StringIO(text.lstrip('\ufeff'))));require(rows and 'smiles' in rows[0],'CSV must contain a smiles column.')
 require(len(rows)<=200,'Upload at most200 compounds per batch.')
 result=[]
 for i,r in enumerate(rows,1):
  result.append(dict(sample_id=str(r.get('sample_id') or f'query_{i}').strip(),smiles=str(r.get('smiles') or '').strip()))
 require(len({r['sample_id'] for r in result})==len(result),'sample_id values must be unique.')
 return result

def similarities(A,B):
 A=np.asarray(A,float);B=np.asarray(B,float);dot=A@B.T;den=(A*A).sum(1)[:,None]+(B*B).sum(1)[None,:]-dot
 require((den>0).all(),'Empty or unsupported fingerprint.');return dot/den

def load_bundle(files):
 contract=json.loads((HERE/'contract.json').read_text());require(sys.version_info[:2]==(3,12),'Use Python3.12 to reproduce the fitted models.')
 require(sklearn.__version__==contract['sklearn'] and rdBase.rdkitVersion==contract['rdkit'],'Install the pinned scikit-learn and RDKit versions from requirements.txt.')
 require(sha((HERE/'data/training_membership.csv').read_bytes())==contract['training_metadata_sha256'],'Training metadata checksum mismatch.')
 members=read_csv(HERE/'data/training_membership.csv');bundles={}
 # Authenticate every model before deserializing either joblib.
 for name in MODEL_NAMES:require(name in files and sha(files[name])==contract['model_sha256'][name],'Wrong model file: '+name+'. Use the final15408 artifact; QA models are rejected.')
 for name in MODEL_NAMES:
  b=joblib.load(io.BytesIO(files[name]));m=b['model'];ids=[r['sample_id'] for r in members]
  require(b['QA_smoke'] is False and b['view']=='count' and b['scaler'] is None,'Wrong final model contract.')
  require(b['training_ids']==ids and m.classes_.tolist()==[0,1],'Training ID order or class order differs.')
  require(np.asarray(b['training_counts']).shape==(2130,2048) and np.asarray(b['training_bits']).shape==(2130,2048),'Training feature shape differs.')
  if name=='ET_count_fixed':
   require(isinstance(m,ExtraTreesClassifier) and all(m.get_params()[k]==v for k,v in contract['ET_params'].items()),'ExtraTrees parameters differ.')
   m.n_jobs=1
  else:require(isinstance(m,SVC) and m.C==10 and m.kernel=='precomputed' and m.class_weight=='balanced' and m.fit_status_==0,'SVM parameters differ.')
  bundles[name]=b
 a,b=[bundles[n] for n in MODEL_NAMES]
 require(np.array_equal(a['training_counts'],b['training_counts']) and np.array_equal(a['training_bits'],b['training_bits']),'Models use different training features.')
 counts=np.asarray(a['training_counts']);bits=np.asarray(a['training_bits']);require(np.array_equal(counts>0,bits>0),'Count/bit feature contract differs.')
 controls=read_csv(HERE/'data/replay_queries.csv');B,C,_=encode([r['smiles'] for r in controls]);K=similarities(C,counts)
 es=a['model'].predict_proba(np.log1p(C.astype(float)))[:,1];ss=b['model'].decision_function(K)
 sp=b['model'].predict(K);reference=read_csv(HERE/'data/replay_scores.csv');maxdiff=0.
 for kind,scores,classes in [('ET_count_fixed',es,(es>=.5).astype(int)),('SVM_count_inner_C',ss,sp)]:
  for q,score,cls in zip(controls,scores,classes):
   r=next(r for r in reference if r['sample_id']==q['sample_id'] and r['model']==kind);diff=abs(float(score)-float(r['score']));maxdiff=max(maxdiff,diff)
   require(diff<1e-10 and int(cls)==int(r['predicted_class']),'Prediction replay failed. Do not use this environment for predictions.')
 keys=[]
 for r in members:
  mol=Chem.MolFromSmiles(r['smiles']);require(mol is not None,'Invalid training metadata structure.')
  keys.append(Chem.MolToSmiles(mol,isomericSmiles=False))
 return dict(models=bundles,members=members,training_keys=keys,lock=threading.RLock(),replay_count=36,max_replay_difference=maxdiff,contract=contract)

def predict(state,queries):
 require(0<len(queries)<=200,'Provide1–200 compounds.');valid=[];invalid=[]
 for r in queries:
  try:
   canonical,key=validate_smiles(r['smiles']);valid.append(dict(**r,canonical_smiles=canonical,connectivity_key=key))
  except ValueError as e:invalid.append(dict(**r,error=str(e)))
 if not valid:return [],[],invalid
 B,C,_=encode([r['canonical_smiles'] for r in valid]);a=state['models']['ET_count_fixed'];b=state['models']['SVM_count_inner_C']
 K=similarities(C,a['training_counts']);KB=similarities(B,a['training_bits'])
 with state['lock']:
  es=a['model'].predict_proba(np.log1p(C.astype(float)))[:,1];ss=b['model'].decision_function(K);sp=b['model'].predict(K)
 results=[];neighbors=[]
 for i,r in enumerate(valid):
  ep=int(es[i]>=.5);sv=int(sp[i]);overlaps=[m['sample_id'] for key,m in zip(state['training_keys'],state['members']) if key==r['connectivity_key']]
  ties=np.flatnonzero(np.isclose(K[i],K[i].max(),rtol=0,atol=1e-12))
  results.append(dict(sample_id=r['sample_id'],input_smiles=r['smiles'],canonical_smiles=r['canonical_smiles'],ExtraTrees_class='Active' if ep else 'Inactive',ExtraTrees_uncalibrated_score=float(es[i]),SVM_class='Active' if sv else 'Inactive',SVM_decision_margin=float(ss[i]),models_agree=ep==sv,max_training_bit_Tanimoto=float(KB[i].max()),max_training_generalized_count_Tanimoto=float(K[i].max()),count_nearest_ties=len(ties),training_connectivity_overlap=bool(overlaps),overlap_training_ids=';'.join(overlaps),scope='Predicted operational DPPH category for the supplied structure; not measured IC50, confirmed GCMS identity, extract or animal efficacy. Similarities and model agreement are descriptive, not calibrated confidence.'))
  order=np.argsort(-K[i],kind='stable')[:5]
  for rank,j in enumerate(order,1):
   m=state['members'][j];neighbors.append(dict(sample_id=r['sample_id'],rank=rank,training_id=m['sample_id'],generalized_count_Tanimoto=float(K[i,j]),bit_Tanimoto=float(KB[i,j]),reported_training_class='Active' if int(m['label']) else 'Inactive',training_smiles=m['smiles'],source_DOI=m['source_DOI']))
 return results,neighbors,invalid

def csv_bytes(rows):
 if not rows:return b''
 f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return ('\ufeff'+f.getvalue()).encode('utf-8')
