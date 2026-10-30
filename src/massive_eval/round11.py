#!/usr/bin/env python3
"""Read-only paired reanalysis. No model loading, fitting, inference, or selection."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'4')
import argparse, csv, hashlib, itertools, json, platform, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import binomtest

LOCALES = ('en-US','zh-CN','ja-JP')
FAMILIES = ('char_lr','minilm_shared_lr')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def readcsv(p):
    with Path(p).open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))
def writecsv(p,rows):
    rows=list(rows)
    keys=list(dict.fromkeys(k for row in rows for k in row if not k.startswith('_')))
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        w.writerows({k:v for k,v in row.items() if not k.startswith('_')} for row in rows)
def model_name(family,locale): return family+'_'+locale if family=='char_lr' else family
def family_name(model): return 'char_lr' if model.startswith('char_lr_') else model

def scores(p):
    ordered=np.sort(p,axis=1)
    return {'confidence':ordered[:,-1], 'margin':ordered[:,-1]-ordered[:,-2],
            'negative_entropy':np.sum(p*np.log(np.maximum(p,1e-300)),axis=1)/np.log(p.shape[1])}
def unique_index(rows,key):
    out={}
    for r in rows:
        k=key(r)
        if k in out: raise ValueError(f'duplicate key {k}')
        out[k]=r
    return out

def safe_ratio(n,d):
    n,d=np.broadcast_arrays(np.asarray(n,dtype=float),np.asarray(d,dtype=float))
    return np.divide(n,d,out=np.full(n.shape,np.nan),where=d!=0)
def events(accept,correct):
    bad=accept & ~correct
    any_bad=bad.any(axis=1)
    return {'decision_flip': accept.any(axis=1)&~accept.all(axis=1),
            'any_accepted_error':any_bad,
            'mixed_accepted_error':any_bad&~bad.all(axis=1),
            'accepted_error_and_other_correct':any_bad&correct.any(axis=1),
            'accepted_error_and_other_abstains':any_bad&(~accept).any(axis=1),
            'all_three_accept_error':bad.all(axis=1)}
def holm(ps):
    order=np.argsort(ps); adjusted=np.empty(len(ps));level=0.
    for j,i in enumerate(order):
        level=max(level,(len(ps)-j)*ps[i]);adjusted[i]=min(1.,level)
    return adjusted

def snapshot(root):
    return {str(p.relative_to(root)):{'sha256':sha(p),'size_bytes':p.stat().st_size}
            for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}

def verify_manifest(root):
    manifest=json.loads((root/'delivery_manifest.json').read_text())
    files=manifest['files']
    for item in files:
        assert sha(root/item['path'])==item['sha256'], f'manifest mismatch {root/item["path"]}'
    return len(files)

class Registry:
    def __init__(self,n):
        self.n=n;self.vectors=[];self.keys={};self.one=self.add(np.ones(n))
    def add(self,v):
        v=np.asarray(v,dtype=np.float64)
        assert v.shape==(self.n,) and np.isfinite(v).all()
        k=v.tobytes()
        if k not in self.keys: self.keys[k]=len(self.vectors);self.vectors.append(v)
        return self.keys[k]
    def fraction(self,row,name,num,den=None):
        a=self.add(num);b=self.one if den is None else self.add(den)
        row[name+'_numerator']=int(np.sum(self.vectors[a])); row[name+'_denominator']=int(np.sum(self.vectors[b]))
        d=row[name+'_denominator'];row[name]=row[name+'_numerator']/d if d else None
        row.setdefault('_ratios',{})[name]=(a,b)

def bootstrap_counts(matrix,groups,b,seed):
    unique,inv=np.unique(groups,return_inverse=True)
    aggregate=np.zeros((len(unique),matrix.shape[1]),dtype=float)
    np.add.at(aggregate,inv,matrix)
    rng=np.random.default_rng(seed)
    result=np.empty((b,matrix.shape[1]),dtype=float)
    for start in range(0,b,250):
        k=min(250,b-start)
        weights=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=k).astype(float)
        result[start:start+k]=weights@aggregate
    return result

def bounds(x,alpha=.05):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return [float(v) for v in np.quantile(x,[alpha/2,1-alpha/2])] if len(x) else [None,None]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=Path(__file__).with_name('analysis_config.json'));ap.add_argument('--output',type=Path)
    args=ap.parse_args();cfg=json.loads(args.config.read_text());out=(args.output or args.config.parent).resolve();out.mkdir(parents=True,exist_ok=True)
    (out/'provenance').mkdir(exist_ok=True)
    src=(args.config.parent/cfg['input_directory']).resolve();r9=(args.config.parent/cfg['round9_directory']).resolve()
    t0=time.perf_counter();checks=[]
    def check(name,condition,detail=None):
        assert condition,name;checks.append({'check':name,'pass':True,'detail':detail})
    before={'round10':snapshot(src),'round9':snapshot(r9)}
    dump(out/'provenance/input_manifest.json',{'absolute_roots':{'round10':str(src),'round9':str(r9)},'files':before})
    check('round10_manifest',True,verify_manifest(src));check('round9_manifest',True,verify_manifest(r9))
    lock=json.loads((src/'dev_lock.json').read_text())
    for file,h in lock['artifacts'].items(): check('dev_lock:'+file,sha(src/file)==h)
    raw=readcsv(src/'test_predictions.csv')
    idx=unique_index(raw,lambda r:(family_name(r['model']),r['locale'],r['id']))
    ids_by={(f,l):{i for ff,ll,i in idx if (ff,ll)==(f,l)} for f in FAMILIES for l in LOCALES}
    common_ids=set.intersection(*ids_by.values());union_ids=set.union(*ids_by.values());ids=sorted(common_ids,key=int);n=len(ids)
    check('same_test_ids_for_all_six_predictions',common_ids==union_ids,{'intersection':n,'union':len(union_ids),'per_model_language':{f+'|'+l:len(s) for (f,l),s in ids_by.items()}})
    membership=readcsv(src/'split_membership.csv')
    expected={r['id'] for r in membership if r['partition']=='test' and r['retain']=='True'}
    check('test_ids_equal_frozen_retained_membership',set(ids)==expected)
    retained=[r for r in membership if r['retain']=='True']
    for field in ['id','group_id']:
        by={p:{r[field] for r in retained if r['partition']==p} for p in ['train','dev','test']}
        check('no_split_overlap_'+field,all(not (by[a]&by[b]) for a,b in itertools.combinations(by,2)))
    groups=np.array([idx[(FAMILIES[0],LOCALES[0],i)]['group_id'] for i in ids])
    gold=np.array([idx[(FAMILIES[0],LOCALES[0],i)]['gold_intent'] for i in ids])
    check('gold_labels_and_group_ids_aligned',all(idx[f,l,i]['gold_intent']==gold[j] and idx[f,l,i]['group_id']==groups[j] for j,i in enumerate(ids) for f in FAMILIES for l in LOCALES))
    pred={};correct={};test_scores={};dev_scores={};dev_ids_by={}
    for f in FAMILIES:
        for l in LOCALES:
            rr=[idx[f,l,i] for i in ids];pred[f,l]=np.array([r['predicted_intent'] for r in rr]);correct[f,l]=pred[f,l]==gold
            check('correct_flags:'+f+l,all((r['correct']=='True')==v for r,v in zip(rr,correct[f,l])))
            for split in ['test','dev']:
                p=src/f'{split}_probabilities_{model_name(f,l)}_{l}.npz'
                with np.load(p,allow_pickle=False) as z:
                    zi=z['ids'].astype(str); ix={i:j for j,i in enumerate(zi)}
                    check('unique_npz_ids:'+p.name,len(ix)==len(zi))
                    target_ids=ids if split=='test' else sorted(ix,key=int)
                    if split=='test':check('npz_ids_match_csv:'+p.name,set(ix)==set(ids))
                    order=[ix[i] for i in target_ids];pr=z['probabilities'][order];classes=z['classes'].astype(str);labels=z['labels'][order]
                    check('valid_probabilities:'+p.name,np.isfinite(pr).all() and (pr>=0).all() and np.allclose(pr.sum(axis=1),1,atol=1e-6))
                    ss=scores(pr)
                    if split=='test':
                        check('npz_predictions_and_labels:'+p.name,np.array_equal(classes[pr.argmax(axis=1)],pred[f,l]) and np.array_equal(classes[labels],gold))
                        for strategy in cfg['strategies']:
                            check('saved_scores:'+f+l+strategy,np.allclose(ss[strategy],[float(r[strategy]) for r in rr],rtol=1e-10,atol=1e-12))
                        test_scores[f,l]=ss
                    else: dev_scores[f,l]=ss;dev_ids_by[f,l]=target_ids
    check('dev_ids_aligned',len({tuple(v) for v in dev_ids_by.values()})==1)
    check('dev_test_ids_disjoint',not set(next(iter(dev_ids_by.values())))&set(ids))
    thresholds=json.loads((src/'dev_thresholds.json').read_text())
    tidx=unique_index(thresholds,lambda r:(family_name(r['model']),r['locale'],r['strategy'],float(r['target_dev_coverage'])))
    expected_keys=set(itertools.product(FAMILIES,LOCALES,cfg['strategies'],cfg['dev_coverage_targets']))
    check('all_108_thresholds_present',set(tidx)==expected_keys,len(tidx))
    check('all_thresholds_selected_on_dev',all(r['selected_on']=='dev' for r in thresholds))
    reg=Registry(n);riskrows=[];paired=[];langbase=[];correctnesspatterns=[];detail=[];group_rows=[]
    ridx={};accepts={};difficulty_cuts={};difficulty={}
    old=unique_index(readcsv(src/'selective_results.csv'),lambda r:(family_name(r['model']),r['locale'],r['strategy'],float(r['target_dev_coverage'])))
    for f in FAMILIES:
        devmean=np.mean([dev_scores[f,l]['margin'] for l in LOCALES],axis=0)
        cuts=np.quantile(devmean,[1/3,2/3]);difficulty_cuts[f]=cuts.tolist()
        testmean=np.mean([test_scores[f,l]['margin'] for l in LOCALES],axis=0)
        difficulty[f]=np.array(['low_margin','mid_margin','high_margin'])[np.searchsorted(cuts,testmean,side='right')]
        pp=np.column_stack([pred[f,l] for l in LOCALES]);cc=np.column_stack([correct[f,l] for l in LOCALES])
        base={'family':f,'locale_a':'all_three','locale_b':'all_three','n':n}
        reg.fraction(base,'prediction_agreement',(pp==pp[:,[0]]).all(axis=1));reg.fraction(base,'correctness_disagreement',cc.any(axis=1)&~cc.all(axis=1));reg.fraction(base,'all_correct',cc.all(axis=1));reg.fraction(base,'all_wrong',(~cc).all(axis=1));langbase.append(base)
        for a,b in itertools.combinations(range(3),2):
            row={'family':f,'locale_a':LOCALES[a],'locale_b':LOCALES[b],'n':n}
            reg.fraction(row,'prediction_agreement',pp[:,a]==pp[:,b]);reg.fraction(row,'correctness_disagreement',cc[:,a]!=cc[:,b]);reg.fraction(row,'a_correct_b_wrong',cc[:,a]&~cc[:,b]);reg.fraction(row,'a_wrong_b_correct',~cc[:,a]&cc[:,b]);langbase.append(row)
        for pat in itertools.product([0,1],repeat=3):
            count=int((cc==np.array(pat)).all(axis=1).sum());correctnesspatterns.append({'family':f,'correct_pattern_en_zh_ja':''.join(map(str,pat)),'numerator':count,'denominator':n,'proportion':count/n})
        for j,i in enumerate(ids):
            row={'family':f,'id':i,'group_id':groups[j],'gold_intent':gold[j],'difficulty':difficulty[f][j]}
            for k,l in enumerate(LOCALES): row.update({l+'_predicted':pp[j,k],l+'_correct':bool(cc[j,k])})
            detail.append(row)
        for strategy,target in itertools.product(cfg['strategies'],cfg['dev_coverage_targets']):
            ac=[]
            for l in LOCALES:
                key=f,l,strategy,target;th=tidx[key]['threshold'];accept=test_scores[f,l][strategy]>=th;error=~correct[f,l];ae=accept&error
                accepts[key]=accept;ac.append(accept)
                row={'family':f,'locale':l,'strategy':strategy,'target_dev_coverage':target,'threshold':th,'n':n,'accepted':int(accept.sum()),'accepted_errors':int(ae.sum()),'original_errors':int(error.sum()),'correct_abstained':int((~accept&~error).sum()),'errors_abstained':int((~accept&error).sum())}
                for name,num,den in [('accuracy',~error,None),('coverage',accept,None),('accepted_risk',ae,accept),('error_abstention',~accept&error,error),('correct_abstention',~accept&~error,~error),('accepted_error_population_rate',ae,None),('error_acceptance',ae,error),('correct_acceptance',accept&~error,~error)]:reg.fraction(row,name,num,den)
                ridx[key]=row;riskrows.append(row)
                check('recomputed_frozen_result:'+str(key),row['accepted']==int(old[key]['test_accepted']) and row['accepted_errors']==int(old[key]['test_errors_accepted']) and abs(row['accepted_risk']-float(old[key]['test_risk']))<1e-12)
                dev_accept=dev_scores[f,l][strategy]>=th
                check('dev_acceptance_replay:'+str(key),int(dev_accept.sum())==tidx[key]['dev_accepted'])
            aa=np.column_stack(ac);ev=events(aa,cc)
            row={'family':f,'strategy':strategy,'target_dev_coverage':target,'pair':'all_three','n':n}
            for name,v in ev.items():reg.fraction(row,name,v)
            reg.fraction(row,'mixed_given_any_accepted_error',ev['mixed_accepted_error'],ev['any_accepted_error']);paired.append(row)
            for a,b in itertools.combinations(range(3),2):
                row={'family':f,'strategy':strategy,'target_dev_coverage':target,'pair':LOCALES[a]+'|'+LOCALES[b],'n':n}
                reg.fraction(row,'decision_flip',aa[:,a]!=aa[:,b]);reg.fraction(row,'a_accepts_error_b_correct_or_abstains',aa[:,a]&~cc[:,a]&(cc[:,b]|~aa[:,b]));reg.fraction(row,'b_accepts_error_a_correct_or_abstains',aa[:,b]&~cc[:,b]&(cc[:,a]|~aa[:,a]));paired.append(row)
            for j,i in enumerate(ids):
                for k,l in enumerate(LOCALES):detail[(0 if f=='char_lr' else n)+j][f'{strategy}_{target:g}_{l}_accept']=bool(aa[j,k])
            masks=[('intent',g,gold==g) for g in sorted(set(gold))]+[('dev_margin_tercile',g,difficulty[f]==g) for g in ['low_margin','mid_margin','high_margin']]
            for flag in ['slot_mismatch_168','normalized_label_conflict']:
                mask=np.array([any(idx[f,l,i][flag]=='True' for l in LOCALES) for i in ids])
                masks.extend((flag,str(b),mask==b) for b in [False,True])
            for axis,group,mask in masks:
                gn=int(mask.sum())
                if not gn:continue
                row={'family':f,'strategy':strategy,'target_dev_coverage':target,'axis':axis,'group':group,'locale':'all_three','n':gn,'prediction_agreement_count':int(((pp==pp[:,[0]]).all(axis=1)&mask).sum()),'correctness_disagreement_count':int((cc.any(axis=1)&~cc.all(axis=1)&mask).sum()),'decision_flip_count':int((ev['decision_flip']&mask).sum()),'mixed_accepted_error_count':int((ev['mixed_accepted_error']&mask).sum())};group_rows.append(row)
                for k,l in enumerate(LOCALES):
                    acn=int(aa[mask,k].sum());ae=int((aa[mask,k]&~cc[mask,k]).sum());cor=int(cc[mask,k].sum());err=gn-cor
                    group_rows.append({'family':f,'strategy':strategy,'target_dev_coverage':target,'axis':axis,'group':group,'locale':l,'n':gn,'correct':cor,'accuracy':cor/gn,'accepted':acn,'accepted_errors':ae,'accepted_risk':ae/acn if acn else None,'coverage':acn/gn,'original_errors':err,'errors_abstained':err-ae,'correct_abstained':cor-(acn-ae)})
    print(f'Validated {len(raw)} rows, {n} common test IDs, {len(set(groups))} components, {len(tidx)} thresholds.',flush=True)
    matrix=np.column_stack(reg.vectors);boot={}
    for name,clusters,seed in [('id',np.array(ids),cfg['bootstrap_seed']),('component',groups,cfg['bootstrap_seed']+1)]:
        print(f'Bootstrap {name}: {cfg["bootstrap_replicates"]} replicates, {matrix.shape[1]} sufficient statistics.',flush=True)
        boot[name]=bootstrap_counts(matrix,clusters,cfg['bootstrap_replicates'],seed)
    for row in riskrows+paired+langbase:
        for name,(a,b) in row['_ratios'].items():
            for unit,bb in boot.items():
                ci=bounds(safe_ratio(bb[:,a],bb[:,b]));row[name+'_'+unit+'_ci_low'],row[name+'_'+unit+'_ci_high']=ci
    def reps(row,metric,unit):
        a,b=row['_ratios'][metric];return safe_ratio(boot[unit][:,a],boot[unit][:,b])
    diffs=[];worst=[];standard=[]
    for f,strategy,target in itertools.product(FAMILIES,cfg['strategies'],cfg['dev_coverage_targets']):
        rows=[ridx[f,l,strategy,target] for l in LOCALES]
        er=np.array([1-r['accuracy'] for r in rows]);eref=float(er.mean())
        sr=[];srep={u:[] for u in boot}
        for l,r in zip(LOCALES,rows):
            a=r['error_acceptance'];b=r['correct_acceptance'];value=eref*a/(eref*a+(1-eref)*b);sr.append(value)
            row={'family':f,'strategy':strategy,'target_dev_coverage':target,'locale':l,'common_error_prevalence':eref,'standardized_risk':value,'interpretation':'algebraic common-prevalence standardization; not causal or a new test set'}
            for u in boot:
                eb=np.mean([1-reps(x,'accuracy',u) for x in rows],axis=0);ab=reps(r,'error_acceptance',u);cb=reps(r,'correct_acceptance',u)
                vv=safe_ratio(eb*ab,eb*ab+(1-eb)*cb);srep[u].append(vv);row['standardized_risk_'+u+'_ci_low'],row['standardized_risk_'+u+'_ci_high']=bounds(vv)
            standard.append(row)
        risks=[r['accepted_risk'] for r in rows];wi=int(np.argmax(risks));wr={'family':f,'strategy':strategy,'target_dev_coverage':target,'observed_worst_locale':LOCALES[wi],'worst_language_risk':risks[wi],'accepted_errors':rows[wi]['accepted_errors'],'accepted':rows[wi]['accepted']}
        for u in boot:wr['worst_risk_'+u+'_ci_low'],wr['worst_risk_'+u+'_ci_high']=bounds(np.max([reps(r,'accepted_risk',u) for r in rows],axis=0))
        worst.append(wr)
        for a,b in itertools.combinations(range(3),2):
            ra,rb=rows[a],rows[b];row={'family':f,'strategy':strategy,'target_dev_coverage':target,'locale_a':LOCALES[a],'locale_b':LOCALES[b],'direction':'a minus b','risk_difference':ra['accepted_risk']-rb['accepted_risk'],'a_accepted_errors':ra['accepted_errors'],'a_accepted':ra['accepted'],'b_accepted_errors':rb['accepted_errors'],'b_accepted':rb['accepted'],'base_error_difference':er[a]-er[b],'standardized_risk_difference':sr[a]-sr[b]}
            for u in boot:
                row['risk_difference_'+u+'_ci_low'],row['risk_difference_'+u+'_ci_high']=bounds(reps(ra,'accepted_risk',u)-reps(rb,'accepted_risk',u))
                row['standardized_difference_'+u+'_ci_low'],row['standardized_difference_'+u+'_ci_high']=bounds(srep[u][a]-srep[u][b])
            diffs.append(row)
    models=[]
    for l in LOCALES:
        ca=correct['char_lr',l];cb=correct['minilm_shared_lr',l];aonly=int((ca&~cb).sum());bonly=int((~ca&cb).sum());nd=aonly+bonly
        p=float(binomtest(bonly,nd,.5).pvalue) if nd else 1.
        row={'locale':l,'n_paired':n,'n_char_available':len(ids_by['char_lr',l]),'n_encoder_available':len(ids_by['minilm_shared_lr',l]),'excluded_unpaired_char':len(ids_by['char_lr',l]-common_ids),'excluded_unpaired_encoder':len(ids_by['minilm_shared_lr',l]-common_ids),'char_correct':int(ca.sum()),'encoder_correct':int(cb.sum()),'both_correct':int((ca&cb).sum()),'both_wrong':int((~ca&~cb).sum()),'char_only_correct':aonly,'encoder_only_correct':bonly,'discordant':nd,'accuracy_difference_encoder_minus_char':float(cb.mean()-ca.mean()),'mcnemar_exact_p':p,'mcnemar_assumption':'source IDs independent; duplicate-component dependence examined by bootstrap, not by this exact p value'}
        for u,bb in boot.items():
            delta=safe_ratio(bb[:,reg.add(cb)]-bb[:,reg.add(ca)],bb[:,reg.one])
            row['difference_'+u+'_ci_low'],row['difference_'+u+'_ci_high']=bounds(delta)
            row['difference_'+u+'_bonferroni_ci_low'],row['difference_'+u+'_bonferroni_ci_high']=bounds(delta,.05/3)
        models.append(row)
    for r,p in zip(models,holm([r['mcnemar_exact_p'] for r in models])):r['mcnemar_holm_p']=float(p)
    outputs={'risk_coverage_summary.csv':riskrows,'paired_language_results.csv':paired,'language_prediction_agreement.csv':langbase,'correctness_patterns.csv':correctnesspatterns,'paired_model_comparison.csv':models,'cross_language_risk_differences.csv':diffs,'worst_language_risk.csv':worst,'standardized_risk_diagnostic.csv':standard,'group_diagnostics.csv':group_rows,'paired_test_id_details.csv':detail}
    for name,rows in outputs.items():writecsv(out/name,rows)
    after={'round10':snapshot(src),'round9':snapshot(r9)};check('all_historical_files_unchanged',before==after)
    dump(out/'provenance/integrity_after.json',{'unchanged':before==after,'round10_file_count':len(after['round10']),'round9_file_count':len(after['round9'])})
    summary={'n_common_ids':n,'n_source_text_components':len(set(groups)),'n_prediction_rows':len(raw),'n_thresholds':len(tidx),'n_intents_in_test':len(set(gold)),'test_ids':ids,'difficulty_dev_margin_tercile_cuts':difficulty_cuts,'bootstrap_replicates':cfg['bootstrap_replicates'],'bootstrap_seed':cfg['bootstrap_seed'],'checks_passed':len(checks),'output_rows':{k:len(v) for k,v in outputs.items()},'analysis_scope':cfg['analysis_status'],'historical_assets_unchanged':before==after}
    dump(out/'analysis_summary.json',summary);dump(out/'verification_results.json',checks)
    dump(out/'environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'elapsed_seconds':time.perf_counter()-t0,'paid_api_calls':0,'new_model_training':0,'new_predictions':0,'new_human_annotations':0,'script_sha256':sha(__file__),'config_sha256':sha(args.config)})
    print(json.dumps({k:v for k,v in summary.items() if k!='test_ids'},ensure_ascii=False,indent=2),flush=True)
    print(f'Elapsed seconds: {time.perf_counter()-t0:.3f}',flush=True)
if __name__=='__main__':main()
