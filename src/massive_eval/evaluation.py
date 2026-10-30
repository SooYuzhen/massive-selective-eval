"""Reuse the frozen metric definitions; no threshold fitting on test."""
import csv,itertools,json
from pathlib import Path
import numpy as np
from .core import selective,curve,write_csv
from .round11 import unique_index,events,family_name

def read_predictions(path):
    with Path(path).open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    unique_index(rows,lambda r:(r['model'],r['locale'],r['id']))
    for r in rows:
        if not r.get('gold_intent') or not r.get('predicted_intent'):raise ValueError('Evaluation requires reference and predicted labels')
        for k in ['confidence','margin','negative_entropy']:
            if not np.isfinite(float(r[k])):raise ValueError('Nonfinite prediction score')
    return rows

def grouped(rows):
    groups={}
    for r in rows:groups.setdefault((r['model'],r['locale']),[]).append(r)
    return groups

def evaluate(predictions,thresholds,out):
    rows=read_predictions(predictions);groups=grouped(rows);ts=json.loads(Path(thresholds).read_text());result=[]
    unique_index(ts,lambda t:(t['model'],t['locale'],t['strategy'],float(t['target_dev_coverage'])))
    for t in ts:
        if t.get('selected_on')!='dev':raise ValueError('Only dev-frozen thresholds are accepted')
        rr=groups[t['model'],t['locale']];correct=np.array([r['gold_intent']==r['predicted_intent'] for r in rr]);score=np.array([float(r[t['strategy']]) for r in rr]);m=selective(correct,score,float(t['threshold']))
        keep=score>=float(t['threshold']);error=~correct
        result.append({'model':t['model'],'locale':t['locale'],'strategy':t['strategy'],'target_dev_coverage':t['target_dev_coverage'],'threshold':t['threshold'],'n':len(rr),**m,'errors_total':int(error.sum()),'errors_abstained':int((error&~keep).sum()),'correct_total':int(correct.sum()),'correct_abstained':int((correct&~keep).sum())})
    write_csv(out,result);return result

def curves(predictions,out):
    result=[]
    for (model,locale),rs in grouped(read_predictions(predictions)).items():
        correct=np.array([r['gold_intent']==r['predicted_intent'] for r in rs])
        for strategy in ['confidence','margin','negative_entropy']:
            cover,risk,aurc=curve(correct,[float(r[strategy]) for r in rs])
            for c,r in zip(cover,risk):result.append({'model':model,'locale':locale,'strategy':strategy,'coverage':float(c),'risk':float(r),'aurc_right_step':aurc,'role':'descriptive_rank_curve_not_threshold_selection'})
    write_csv(out,result);return result

def paired(predictions,thresholds,out,locales=('en-US','zh-CN','ja-JP')):
    rows=read_predictions(predictions);index=unique_index(rows,lambda r:(family_name(r['model']),r['locale'],r['id']))
    ts=json.loads(Path(thresholds).read_text());ti=unique_index(ts,lambda t:(family_name(t['model']),t['locale'],t['strategy'],float(t['target_dev_coverage'])))
    if any(t.get('selected_on')!='dev' for t in ts):raise ValueError('Thresholds must be frozen on dev')
    result=[]
    for family in sorted({family_name(r['model']) for r in rows}):
        sets=[{i for f,l,i in index if f==family and l==locale} for locale in locales]
        common=set.intersection(*sets);ids=sorted(common);union=set.union(*sets)
        if not ids:raise ValueError('No common source IDs')
        pp=np.array([[index[family,l,i]['predicted_intent'] for l in locales] for i in ids]);gold=[index[family,locales[0],i]['gold_intent'] for i in ids]
        if any(index[family,l,i]['gold_intent']!=g for i,g in zip(ids,gold) for l in locales):raise ValueError('Paired reference label conflict')
        cc=pp==np.array(gold)[:,None]
        settings=sorted({(s,t) for f,l,s,t in ti if f==family})
        for strategy,target in settings:
            aa=np.array([[float(index[family,l,i][strategy])>=float(ti[family,l,strategy,target]['threshold']) for l in locales] for i in ids]);ev=events(aa,cc)
            row={'family':family,'strategy':strategy,'target_dev_coverage':target,'common_ids':len(ids),'union_ids':len(union),'excluded_noncommon_ids':len(union-common),'prediction_agreement_count':int((pp==pp[:,[0]]).all(axis=1).sum()),'correctness_disagreement_count':int((cc.any(axis=1)&~cc.all(axis=1)).sum())}
            for k,v in ev.items():row[k+'_numerator']=int(v.sum());row[k+'_denominator']=len(ids);row[k]=float(v.mean())
            result.append(row)
    write_csv(out,result);return result
