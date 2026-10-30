"""Compare local CLI replay outputs with user-supplied historical references."""
import argparse,csv,hashlib,json
from pathlib import Path

def rows(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--round10',type=Path,required=True);p.add_argument('--round11',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);a=p.parse_args();checks=0
 key=lambda r:(r['model'],r['locale'],r['strategy'],float(r['target_dev_coverage']))
 new={key(r):r for r in rows(a.replay/'selective.csv')}
 for r in rows(a.round10/'selective_results.csv'):
  x=new[key(r)];assert int(x['accepted'])==int(r['test_accepted']);assert int(x['errors_accepted'])==int(r['test_errors_accepted']);assert abs(float(x['risk'])-float(r['test_risk']))<1e-12;checks+=1
 old={(r['family'],r['strategy'],r['target_dev_coverage']):r for r in rows(a.round11/'paired_language_results.csv') if r['pair']=='all_three'}
 for r in rows(a.replay/'paired.csv'):
  x=old[r['family'],r['strategy'],r['target_dev_coverage']];assert all(v==x[k] for k,v in r.items() if k.endswith('_numerator') or k.endswith('_denominator'));checks+=1
 curves={ (r['model'],r['locale'],r['strategy']):float(r['aurc_right_step']) for r in rows(a.replay/'curves.csv')}
 for r in rows(a.round10/'test_summary.csv'):
  for s in ['confidence','margin','negative_entropy']:assert abs(curves[r['model'],r['locale'],s]-float(r['aurc_'+s]))<1e-12;checks+=1
 names=json.loads((a.round11/'analysis_summary.json').read_text())['output_rows']
 for name in names:assert sha(a.replay/'round11_replay'/name)==sha(a.round11/name);checks+=1
 print(json.dumps({'status':'PASS','numerical_checks':checks,'training':False,'inference':False}))
if __name__=='__main__':main()
