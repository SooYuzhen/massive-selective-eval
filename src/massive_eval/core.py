import csv,hashlib,json,re,unicodedata
from pathlib import Path
import numpy as np
LOCALES=('en-US','zh-CN','ja-JP')
def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write_csv(path,rows,fields=None):
 rows=list(rows);fields=fields or list(rows[0])
 with open(path,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def normalize(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def slot_types(s):
 from collections import Counter
 return Counter(x.strip() for x in re.findall(r'\[([^:\[\]]+)\s*:',s))
class DSU:
 def __init__(self,ids):self.p={x:x for x in ids}
 def find(self,x):
  if self.p[x]!=x:self.p[x]=self.find(self.p[x])
  return self.p[x]
 def union(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.p[max(a,b,key=lambda x:int(x))]=min(a,b,key=lambda x:int(x))
def scores(p):
 p=np.asarray(p);ordered=np.sort(p,axis=1)
 return {'confidence':ordered[:,-1],'margin':ordered[:,-1]-ordered[:,-2],
         'negative_entropy':np.sum(p*np.log(np.maximum(p,1e-300)),axis=1)/np.log(p.shape[1])}
def threshold_at_coverage(score,target):
 if not 0<target<=1:raise ValueError('coverage outside (0,1]')
 if target==1:return -1e100
 return float(np.sort(score)[::-1][int(np.ceil(target*len(score)))-1])
def selective(correct,score,threshold):
 keep=np.asarray(score)>=threshold;n=int(keep.sum())
 return {'accepted':n,'abstained':int(len(keep)-n),'coverage':float(keep.mean()),
         'risk':float(1-np.asarray(correct)[keep].mean()) if n else None,
         'accuracy_accepted':float(np.asarray(correct)[keep].mean()) if n else None,
         'errors_accepted':int((~np.asarray(correct,dtype=bool)[keep]).sum())}
def curve(correct,score):
 # Include complete tie groups. No arbitrary tie ordering or test label tie-breaking.
 order=np.argsort(-np.asarray(score),kind='stable');sc=np.asarray(score)[order];err=1-np.asarray(correct,dtype=float)[order]
 ends=np.r_[np.flatnonzero(sc[:-1]!=sc[1:])+1,len(sc)]
 risks=np.cumsum(err)[ends-1]/ends;cover=ends/len(sc)
 aurc=float(np.sum(risks*np.diff(np.r_[0,cover])))
 return cover,risks,aurc
