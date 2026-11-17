from pathlib import Path
import json,os,sys
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from predictor import MODEL_NAMES,load_bundle,predict,parse_queries,csv_bytes,read_csv
HERE=Path(__file__).resolve().parent
st.set_page_config(page_title='DPPH Molecular Predictor',page_icon='🧪',layout='wide')
st.markdown("""
<style>
.stApp {background: #f4f8fa; color: #173548;}
[data-testid="stHeader"] {background: rgba(244,248,250,.94);}
[data-testid="stSidebar"] {background: #e8f2f4; border-right: 1px solid #d5e6e9;}
[data-testid="stMainBlockContainer"] {max-width: 1180px; padding-top: 2rem;}
h1,h2,h3 {color: #143e51; letter-spacing: -.025em;}
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea {background: #fff; border: 1px solid #cadde2; border-radius: 10px;}
.stButton button[kind="primary"] {background: #087f83; border-color: #087f83; border-radius: 10px; font-weight: 600; padding: .65rem 1.4rem;}
.stButton button[kind="primary"]:hover {background: #06696d; border-color: #06696d;}
[data-testid="stMetric"] {background: white; padding: 1.1rem; border: 1px solid #dce8ec; border-radius: 14px;}
[data-testid="stExpander"] {background: white; border: 1px solid #dce8ec; border-radius: 12px;}
.hero {position:relative; overflow:hidden; background:linear-gradient(115deg,#102f46,#155b69); border-radius:22px; padding:38px 40px; margin:0 0 20px; min-height:245px; color:white;}
.hero-copy {position:relative; z-index:1; max-width:68%;}
.hero-eyebrow {color:#8fe4d6; font-size:11px; letter-spacing:.17em; font-weight:700; margin-bottom:14px;}
.hero h1 {color:white; font-size:clamp(28px,3.5vw,43px); line-height:1.12; margin:0 0 14px; padding:0;}
.hero p {color:#d9ebee; font-size:15px; line-height:1.6; margin:0; max-width:540px;}
.hero-art {position:absolute; width:36%; height:100%; right:0; top:0; opacity:.9; object-fit:cover;}
.endpoint {background:#fff; border-left:4px solid #16a899; border-radius:8px; padding:13px 17px; color:#32515e; font-size:13px; line-height:1.6; margin-bottom:20px;}
@media(max-width:640px) {.hero {padding:28px 24px; min-height:225px;} .hero-copy {max-width:100%;} .hero-art {opacity:.18; width:65%;} [data-testid="stMainBlockContainer"] {padding-top:1rem;}}
</style>
""",unsafe_allow_html=True)
import base64
_banner_svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 300"><defs><radialGradient id="g"><stop stop-color="#52d8c0" stop-opacity=".22"/><stop offset="1" stop-color="#52d8c0" stop-opacity="0"/></radialGradient></defs><circle cx="190" cy="150" r="145" fill="url(#g)"/><g fill="none" stroke="#8ae6d4" stroke-width="2"><path d="M118 103L174 71L230 103V167L174 199L118 167Z"/><path d="M127 109L174 82L221 109M221 116V161L174 188M164 182L127 161V119" opacity=".5"/><path d="M230 103L281 74M230 167L281 197M118 167L68 196M118 103L68 74M174 199V251"/><path d="M281 197L323 173M68 196L26 173" opacity=".5"/></g><g fill="#a9f0de"><circle cx="174" cy="71" r="6"/><circle cx="230" cy="103" r="6"/><circle cx="230" cy="167" r="6"/><circle cx="174" cy="199" r="6"/><circle cx="118" cy="167" r="6"/><circle cx="118" cy="103" r="6"/></g><g fill="#fff" font-family="sans-serif" font-size="18" text-anchor="middle"><text x="286" y="77">OH</text><text x="174" y="271">OH</text></g><g fill="#7edacb" opacity=".5"><circle cx="59" cy="65" r="5"/><circle cx="288" cy="204" r="5"/><circle cx="325" cy="171" r="4"/><circle cx="26" cy="171" r="4"/></g><g fill="#c1eee8" opacity=".25"><circle cx="318" cy="36" r="3"/><circle cx="32" cy="117" r="3"/><circle cx="307" cy="266" r="3"/><circle cx="89" cy="257" r="3"/></g></svg>'
_banner_url='data:image/svg+xml;base64,'+base64.b64encode(_banner_svg.encode()).decode()
st.markdown(f'<section class="hero"><img class="hero-art" src="{_banner_url}" alt="Decorative phenolic molecular structure"/><div class="hero-copy"><div class="hero-eyebrow">MOLECULAR MACHINE LEARNING</div><h1>DPPH Molecular Predictor</h1><p>Explore molecular DPPH activity with ExtraTrees and Tanimoto SVM. Enter a structure, compare predictions, and inspect training similarity.</p></div></section>',unsafe_allow_html=True)
st.markdown('<div class="endpoint"><b>Classification endpoint</b> &nbsp; Active: reported DPPH IC50 &lt;1000 mg/L · Inactive: ≥1000 mg/L.<br>Predictions describe an assumed molecular structure and do not provide a numerical IC50.</div>',unsafe_allow_html=True)

@st.cache_resource(show_spinner='Checking prediction models…')
def verified_models(et_bytes,svm_bytes):return load_bundle(dict(zip(MODEL_NAMES,[et_bytes,svm_bytes])))

with st.sidebar:
 st.caption('DPPH · STRUCTURE TO CATEGORY')
 st.header('Prediction models')
 st.markdown('- ExtraTrees\n- Tanimoto SVM')
 directory=Path(os.environ.get('DPPH_MODEL_DIR',str(HERE/'models')))
 files={}
 for name in MODEL_NAMES:
  local=directory/(name+'.joblib')
  if local.is_file():files[name]=local.read_bytes()

state=None
if len(files)==2:
 try:state=verified_models(files[MODEL_NAMES[0]],files[MODEL_NAMES[1]])
 except Exception as e:st.error(str(e));st.stop()
 st.sidebar.success('Models verified and ready')
else:st.error('Prediction models are temporarily unavailable. Please contact the application maintainer.')

st.subheader('Explore a molecule')
mode=st.radio('Choose input method',['One molecule','Batch CSV','GCMS candidate examples'],horizontal=True)
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
