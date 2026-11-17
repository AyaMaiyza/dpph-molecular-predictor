from pathlib import Path
import json
from predictor import MODEL_NAMES,load_bundle,predict
root=Path(__file__).resolve().parent
state=load_bundle({n:(root/'models'/(n+'.joblib')).read_bytes() for n in MODEL_NAMES})
print(json.dumps(dict(status='release_replay_passed',controls=state['replay_count'],maximum_difference=state['max_replay_difference']),indent=2))
