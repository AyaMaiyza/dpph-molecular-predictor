from pathlib import Path
import json,os,sys
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from predictor import MODEL_NAMES,load_bundle,predict,parse_queries,csv_bytes,read_csv
HERE=Path(__file__).resolve().parent
st.set_page_config(page_title='DPPH Molecular Predictor',page_icon='🧪',layout='wide')
st.title('DPPH Molecular Predictor')
st.write('Enter a molecular structure to predict its DPPH category with two frozen models.')
st.caption('Active: reported DPPH IC50 <1000 mg/L · Inactive: ≥1000 mg/L. The app predicts a category, not a numerical IC50.')

@st.cache_resource(show_spinner='Checking prediction models…')
def verified_models(et_bytes,svm_bytes):return load_bundle(dict(zip(MODEL_NAMES,[et_bytes,svm_bytes])))

with st.sidebar:
 st.header('Prediction models')
 st.markdown('- ExtraTrees\n- Tanimoto SVM')
 directory=Path(os.environ.get('DPPH_MODEL_DIR',str(HERE/'models')))
 files={}
 verification_panel=st.expander('Technical verification',expanded=False)
 with verification_panel:
  st.caption('Frozen count-model artifacts from run15408. No model training runs in this app.')
  for name in MODEL_NAMES:
   local=directory/(name+'.joblib')
   if local.is_file():files[name]=local.read_bytes();st.caption(name+' · file found')
   else:
    upload=st.file_uploader(name,type=['joblib'],key=name)
    if upload:files[name]=upload.getvalue()
  st.caption('Only the recorded final model checksums are accepted.')

state=None
if len(files)==2:
 try:state=verified_models(files[MODEL_NAMES[0]],files[MODEL_NAMES[1]])
 except Exception as e:st.error(str(e));st.stop()
 st.sidebar.success('Models verified and ready')
 with verification_panel:
  st.caption('Model checksums verified ·36/36 reference prediction replays passed.')
  st.caption('These checks confirm reproducible model loading and prediction, not biological accuracy.')
else:st.info('Load both final model files to enable prediction. The download package includes an HPC export script for these files.')

mode=st.radio('Input',['One molecule','Batch CSV','GCMS candidate examples'],horizontal=True)
queries=[];label=''
if mode=='One molecule':
 name=st.text_input('Compound label','query_1')
 smiles=st.text_area('SMILES','CC(C)(C)c1ccc(O)c(C(C)(C)C)c1',height=90)
 queries=[dict(sample_id=name.strip() or 'query_1',smiles=smiles.strip())];label='single molecule'
elif mode=='Batch CSV':
 st.caption('Required column: smiles. Optional column: sample_id. Maximum200 compounds.')
 st.download_button('Download CSV template',b'sample_id,smiles\nDTBP,CC(C)(C)c1ccc(O)c(C(C)(C)C)c1\nDecane,CCCCCCCCCC\n','prediction_template.csv','text/csv')
 file=st.file_uploader('Compound CSV',type=['csv'],key='queries')
 if file:
  try:
   if file.size>2*1024*1024:raise ValueError('CSV limit:2MB.')
   queries=parse_queries(file.getvalue().decode('utf-8-sig'))
  except (ValueError,UnicodeDecodeError) as e:st.error(str(e))
 label='batch'
else:
 controls=read_csv(HERE/'data/replay_queries.csv');names={r['compound_name']:r for r in controls}
 selected=st.multiselect('Choose assumed structures',list(names),default=['2,4-Di-tert-butylphenol'])
 queries=[dict(sample_id=names[x]['sample_id'],smiles=names[x]['smiles']) for x in selected]
 st.caption('GCMS library assignments remain tentative. These examples do not confirm compound identity in the sample.');label='GCMS candidates'

if st.button('Predict DPPH category',type='primary',disabled=state is None or not queries):
 try:
  results,neighbors,errors=predict(state,queries)
  # Persist only the current prediction result, not user input across sessions.
  st.session_state['prediction']=(results,neighbors,errors,label)
 except Exception as e:st.error('Prediction failed: '+str(e))

if 'prediction' in st.session_state:
 results,neighbors,errors,label=st.session_state['prediction']
 st.subheader('Last prediction results · '+label)
 if errors:st.warning(f'{len(errors)} input rows could not be processed.');st.dataframe(pd.DataFrame(errors),hide_index=True)
 if results:
  if len(results)==1:
   r=results[0];a,b,c=st.columns([1,1,1])
   a.metric('ExtraTrees',r['ExtraTrees_class']);b.metric('Tanimoto SVM',r['SVM_class']);c.metric('Model agreement','Agree' if r['models_agree'] else 'Disagree')
   st.image(Draw.MolToImage(Chem.MolFromSmiles(r['canonical_smiles']),size=(430,250)),caption='Encoded input structure')
  shown=[{k:r[k] for k in ['sample_id','ExtraTrees_class','ExtraTrees_uncalibrated_score','SVM_class','SVM_decision_margin','models_agree','max_training_bit_Tanimoto','max_training_generalized_count_Tanimoto','training_connectivity_overlap']} for r in results]
  st.dataframe(pd.DataFrame(shown),hide_index=True)
  if any(r['training_connectivity_overlap'] for r in results):st.info('A supplied structure matches training connectivity. Its prediction is not an independent evaluation.')
  if any(not r['models_agree'] for r in results):st.info('The models disagree for at least one input. Both predictions are retained; no consensus activity is assigned.')
  st.caption('ET score and SVM margin are not calibrated probabilities. Bit and generalized count similarities have different scales; no validated applicability cutoff is assigned.')
  with st.expander('Five nearest training compounds by count similarity'):
   st.dataframe(pd.DataFrame(neighbors),hide_index=True)
  a,b=st.columns(2)
  a.download_button('Download predictions CSV',csv_bytes(results),'DPPH_predictions.csv','text/csv')
  b.download_button('Download nearest neighbors CSV',csv_bytes(neighbors),'DPPH_neighbors.csv','text/csv')
  report=dict(predictions=results,invalid_rows=errors,model_sha256=state['contract']['model_sha256'],model_source_run=15408,replay_controls=state['replay_count'],runtime=dict(python=sys.version.split()[0]),endpoint=state['contract']['endpoint'],scope='Conditional structural classification only; no measured IC50, sample identity confirmation, extract activity or in-vivo efficacy.')
  st.download_button('Download prediction report JSON',json.dumps(report,indent=2),'DPPH_prediction_report.json','application/json')

with st.expander('Model performance and interpretation'):
 st.write('These are existing models, fitted to2130 compounds. Development evaluated263 original compounds using repeatedly inspected component folds. This app provides no new validation.')
 st.table(pd.DataFrame([dict(Model='ExtraTrees counts',Balanced_accuracy='0.722 ±0.039',ROC_AUC='0.789 ±0.058'),dict(Model='Tanimoto SVM counts',Balanced_accuracy='0.723 ±0.055',ROC_AUC='0.807 ±0.064')]))
 st.caption('Mean ±sample SD across five development folds. Similarity and agreement do not establish biological correctness. The app does not predict extract antioxidant activity, acetylcholinesterase inhibition or rat outcomes.')
