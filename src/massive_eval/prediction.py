"""Inference only, with externally supplied, trusted local frozen models."""
import json,os,sys,time
from pathlib import Path
import numpy as np
from . import core

def predict(records,model_path,expected_sha,model_name,out,encoder=None):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    if core.sha(model_path)!=expected_sha:raise ValueError('Model SHA256 mismatch')
    import joblib
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    # Historical vectorizers reference common.normalize; use the identical project function.
    previous=sys.modules.get('common');sys.modules['common']=core
    try:bundle=joblib.load(model_path)
    finally:
        if previous is None:sys.modules.pop('common',None)
        else:sys.modules['common']=previous
    rows=[json.loads(x) for x in Path(records).open() if x.strip()]
    if not rows:raise ValueError('No records')
    seen=set()
    for r in rows:
        key=(r['locale'],str(r['id']))
        if key in seen:raise ValueError('Duplicate source ID within locale')
        seen.add(key)
        if model_name.startswith('char_lr_') and r['locale']!=model_name.removeprefix('char_lr_'):raise ValueError('Wrong locale for per-language head')
    started=time.perf_counter();v=bundle['vectorizer']
    if v is not None:x=v.transform([r['utt'] for r in rows])
    else:
        if encoder is None or not Path(encoder).is_dir():raise ValueError('A local frozen encoder directory is required')
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(4)
        enc=SentenceTransformer(str(encoder),device='cpu',local_files_only=True);enc.max_seq_length=128
        x=enc.encode([r['utt'] for r in rows],batch_size=64,normalize_embeddings=True,convert_to_numpy=True)
    probabilities=bundle['classifier'].predict_proba(x);classes=np.asarray(bundle['classes']);pred=classes[probabilities.argmax(1)];ss=core.scores(probabilities)
    output=[]
    for i,r in enumerate(rows):
        gold=r.get('intent','');output.append({'model':model_name,'locale':r['locale'],'id':str(r['id']),'gold_intent':gold,'predicted_intent':str(pred[i]),'correct':str(pred[i])==gold if gold else '',**{s:float(v[i]) for s,v in ss.items()}})
    core.write_csv(out,output)
    return {'rows':len(rows),'wall_seconds':time.perf_counter()-started,'model_sha256':expected_sha,'training':False,'downloads':False,'note':'Fresh inference output; not a replacement for historical frozen predictions.'}
