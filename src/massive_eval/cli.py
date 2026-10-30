import argparse,json,sys,tempfile
from pathlib import Path
from .core import dump,write_csv,scores,threshold_at_coverage

def new_file(path):
    p=Path(path)
    if p.exists():raise ValueError('Output already exists; choose a new output path')
    p.parent.mkdir(parents=True,exist_ok=True);return p

def demo(out):
    import numpy as np
    from .evaluation import evaluate,curves,paired
    p=Path(out)
    if p.exists() and any(p.iterdir()):raise ValueError('Demo output must be new or empty')
    p.mkdir(parents=True,exist_ok=True);pred=[];ts=[]
    for j,l in enumerate(['en-US','zh-CN','ja-JP']):
        pp=np.array([[.9,.1],[.6,.4],[.3,.7],[.55,.45]])
        if j==1:pp=pp[:,::-1]
        ss=scores(pp)
        for i,gold in enumerate(['red','red','blue','blue']):
            pred.append({'model':'toy','locale':l,'id':str(i),'gold_intent':gold,'predicted_intent':['red','blue'][int(pp[i].argmax())],**{s:float(v[i]) for s,v in ss.items()}})
        for s in ss:ts.append({'model':'toy','locale':l,'strategy':s,'target_dev_coverage':.5,'threshold':float(sorted(ss[s])[2]),'selected_on':'dev','fixture_only':True})
    write_csv(p/'predictions.csv',pred);dump(p/'thresholds.json',ts)
    evaluate(p/'predictions.csv',p/'thresholds.json',p/'selective.csv');curves(p/'predictions.csv',p/'curves.csv');paired(p/'predictions.csv',p/'thresholds.json',p/'paired.csv')
    dump(p/'NOTICE.json',{'synthetic_software_fixture':True,'scientific_evidence':False,'training':False,'dataset_downloads':False})
    return {'status':'PASS','fixture_rows':12,'scientific_evidence':False}

def main():
    a=argparse.ArgumentParser(description='Multilingual Selective Prediction and Reliability Evaluation')
    sub=a.add_subparsers(dest='command',required=True)
    p=sub.add_parser('demo');p.add_argument('--out',required=True)
    p=sub.add_parser('load');p.add_argument('--source',required=True);p.add_argument('--locale',action='append',dest='locales');p.add_argument('--partition',choices=['train','dev','test']);p.add_argument('--membership');p.add_argument('--output',required=True)
    p=sub.add_parser('audit');p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('predict');p.add_argument('--records',required=True);p.add_argument('--model',required=True);p.add_argument('--sha256',required=True);p.add_argument('--model-name',required=True);p.add_argument('--encoder');p.add_argument('--trusted-local-model',action='store_true',required=True,help='Acknowledge that joblib loading executes trusted serialized Python objects');p.add_argument('--output',required=True)
    for command in ['evaluate','curve','paired']:
        p=sub.add_parser(command);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True)
        if command!='curve':p.add_argument('--thresholds',required=True)
    p=sub.add_parser('replay-round11');p.add_argument('--round10',required=True);p.add_argument('--round9',required=True);p.add_argument('--out',required=True);p.add_argument('--bootstrap',type=int,default=5000)
    args=a.parse_args()
    if args.command=='demo':result=demo(args.out)
    elif args.command=='load':
        from .io import load_to_jsonl
        result=load_to_jsonl(args.source,new_file(args.output),args.locales or ['en-US','zh-CN','ja-JP'],args.partition,args.membership)
    elif args.command=='audit':
        from .raw_audit import audit
        audit(args.source,args.out);result={'status':'completed'}
    elif args.command=='predict':
        from .prediction import predict
        result=predict(args.records,args.model,args.sha256,args.model_name,new_file(args.output),args.encoder)
    elif args.command in ['evaluate','curve','paired']:
        from .evaluation import evaluate,curves,paired
        output=new_file(args.output)
        if args.command=='curve':rr=curves(args.predictions,output)
        else:rr={'evaluate':evaluate,'paired':paired}[args.command](args.predictions,args.thresholds,output)
        result={'rows':len(rr),'output':str(output)}
    else:
        from . import round11
        out=Path(args.out).resolve()
        if out.exists() and any(out.iterdir()):raise ValueError('Replay output must be new or empty')
        out.mkdir(parents=True,exist_ok=True)
        cfg={'input_directory':str(Path(args.round10).resolve()),'round9_directory':str(Path(args.round9).resolve()),'strategies':['confidence','margin','negative_entropy'],'dev_coverage_targets':[.5,.7,.8,.9,.95,1.],'bootstrap_replicates':args.bootstrap,'bootstrap_seed':20261030,'analysis_status':'retrospective exploratory reanalysis; fixed historical predictions'}
        dump(out/'analysis_config.json',cfg);sys.argv=[sys.argv[0],'--config',str(out/'analysis_config.json'),'--output',str(out)];round11.main();result={'status':'completed','no_training_or_inference':True}
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
