#!/usr/bin/env python3
"""Read-only MASSIVE 1.1 tar audit. Python stdlib; no network, models or training.
Stream JSONL once; disk-backed SQLite join index is temporary. Raw strings untouched.
Output directory must be new or empty: never overwrites an earlier audit.
"""
import argparse, collections as C, csv, hashlib, itertools, json, platform, re
import sqlite3, sys, tarfile, tempfile, time
from pathlib import Path
from datetime import datetime, timezone

LOCALES = ('en-US','zh-CN','ja-JP')
FIELDS = ('id','locale','partition','utt','annot_utt','intent','scenario','slot_method','judgments')
SLOT = re.compile(r'\[([^:\[\]]+?)\s*:\s*([^\[\]]*)\]')
JDOM = {'intent_score':{0,1,2},'slots_score':{0,1,2},'grammar_score':{0,1,2,3,4},'spelling_score':{0,1,2}}

def js(x): return json.dumps(x, ensure_ascii=False, sort_keys=True)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def dump(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def compact(s): return re.sub(r'\s','',s)
def parse_annotation(text):
    """Strict bracket parser; offsets refer to original annot_utt, not utt."""
    if not isinstance(text,str): return [], '', ['annotation_not_string']
    slots=[]; errors=[]; tail=0
    for m in SLOT.finditer(text):
        if '[' in text[tail:m.start()] or ']' in text[tail:m.start()]: errors.append('unparsed_bracket')
        label=m[1].strip(); value=m[2].strip()
        if not label or not value: errors.append('empty_slot_label_or_value')
        slots.append({'type':label,'value':value,'annot_start':m.start(),'annot_end':m.end()})
        tail=m.end()
    if '[' in text[tail:] or ']' in text[tail:]: errors.append('unparsed_bracket')
    restored=SLOT.sub(lambda m:m[2].strip(),text)
    return slots,restored,sorted(set(errors))
def signature(slots): return C.Counter(s['type'] for s in slots)
def values(slots):
    out=C.defaultdict(list)
    for s in slots: out[s['type']].append(s['value'])
    return {k:sorted(v) for k,v in out.items()}
def pair_flags(a,b):
    # Compare unordered multisets: no row-order or occurrence-position pairing.
    sa,sb=a['slots'],b['slots']
    return {'slot_type_set_equal':set(signature(sa))==set(signature(sb)),
            'slot_type_multiset_equal':signature(sa)==signature(sb),
            'slot_count_equal':len(sa)==len(sb),
            'slot_value_multiset_equal':values(sa)==values(sb)}
def write_csv(path,fields,rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def verify_archive(t):
    names=t.getnames()
    members=[n for n in names if re.fullmatch(r'1\.1/data/[^/]+\.jsonl',n)]
    if len(names)!=len(set(names)): raise ValueError('Duplicate archive member names')
    if '1.1/CHANGELOG.md' not in names: raise ValueError('Missing 1.1 CHANGELOG; version unresolved')
    changelog=t.extractfile('1.1/CHANGELOG.md').read().decode('utf-8')
    if '[1.1]' not in changelog or 'ca-ES' not in changelog or '1.1/data/ca-ES.jsonl' not in names:
        raise ValueError('1.1 evidence missing')
    if len(members)!=52: raise ValueError(f'Expected 52 locale members; found {len(members)}')
    ca=json.loads(t.extractfile('1.1/data/ca-ES.jsonl').readline())
    if ca.get('locale')!='ca-ES': raise ValueError('Catalan marker invalid')
    for loc in LOCALES:
        if f'1.1/data/{loc}.jsonl' not in members: raise ValueError('Missing target locale')
    return members,changelog

def audit(source,out,scratch=None):
    started=time.monotonic();source=Path(source).resolve();out=Path(out).resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Output must be new or empty')
    out.mkdir(parents=True,exist_ok=True);(out/'provenance').mkdir()
    before=sha(source)
    manifest={'source_path':str(source),'size_bytes':source.stat().st_size,'sha256':before,
        'archive_format':'tar (auto-detected by tarfile; local suffix not trusted)',
        'dataset_identity':'MASSIVE original 1.1 release container',
        'remote_checksum_verified':False,'identity_limit':'Local SHA256 fixes byte identity; no publisher checksum was available/compared. 1.1 manifest, Catalan member and original schema verified. Target languages are unchanged from 1.0 per upstream release notes.',
        'upstream_documentation':'https://github.com/alexa/massive', 'members':[]}
    stats={}; anomalies=[]; summary={'evidence_level':'actual_data_audit; not model evaluation or human gold','languages':stats}
    with tempfile.TemporaryDirectory(prefix='massive_index_',dir=scratch) as tmp, tarfile.open(source,'r:*') as t:
        members,changelog=verify_archive(t)
        manifest['version_checks']={'locale_files':len(members),'changelog_1_1':True,'catalan_first_record_locale':'ca-ES','member_inventory':members}
        for name in ('CHANGELOG.md','CITATION.md','NOTICE.md','LICENSE'):
            b=t.extractfile('1.1/'+name).read();(out/'provenance'/name).write_bytes(b)
            manifest['members'].append({'path':'1.1/'+name,'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)})
        con=sqlite3.connect(str(Path(tmp)/'index.sqlite'))
        con.execute('CREATE TABLE rec(loc TEXT,id TEXT,part TEXT,intent TEXT,scenario TEXT,utt TEXT,annot TEXT,slots TEXT,methods TEXT,line INTEGER,rawhash TEXT,payloadhash TEXT,PRIMARY KEY(loc,line))')
        con.execute('CREATE INDEX id_idx ON rec(id,loc)')
        sf=(out/'slot_annotation_audit.csv').open('w',newline='',encoding='utf-8')
        slotfields=['locale','partition','id','source_member','source_line','utt','annot_utt','restored_utt','exact_restore','whitespace_insensitive_restore','parse_errors','slot_count','slots_json','slot_method_json','slot_method_type_set_equal']
        sw=csv.DictWriter(sf,fieldnames=slotfields);sw.writeheader()
        for loc in LOCALES:
            name=f'1.1/data/{loc}.jsonl';h=hashlib.sha256();s={'partitions':{},'field_types':{},'slot_types':C.Counter(),'slot_count_distribution':C.Counter(),'method_values':C.Counter(),'method_slot_types':C.Counter(),'method_item_keysets':C.Counter(),'judgment_length_distribution':C.Counter(),'judgment_keysets':C.Counter(),'judgment_values':{},'judgment_field_types':{},'judgment_missing_keys':C.Counter(),'judgment_disagreement_rows':C.Counter(),'judgment_out_of_domain':C.Counter(),'parse_error_rows':0,'exact_restore_rows':0,'whitespace_insensitive_restore_rows':0,'method_annotation_type_set_mismatch_rows':0,'record_keysets':C.Counter(),'intent_counts':C.Counter(),'scenario_counts':C.Counter(),'physical_lines':0,'valid_json_objects':0,'unexpected_locale_rows':0,'invalid_partition_rows':0}
            stats[loc]=s
            for ln,raw in enumerate(t.extractfile(name),1):
                h.update(raw);s['physical_lines']+=1
                try:r=json.loads(raw)
                except (ValueError,UnicodeError) as e:
                    anomalies.append({'locale':loc,'id':'','line':ln,'kind':'invalid_json','detail':str(e)});continue
                if not isinstance(r,dict):
                    anomalies.append({'locale':loc,'id':'','line':ln,'kind':'non_object','detail':type(r).__name__});continue
                s['valid_json_objects']+=1
                s['record_keysets'][js(sorted(r))]+=1;s['intent_counts'][str(r.get('intent'))]+=1;s['scenario_counts'][str(r.get('scenario'))]+=1
                rid=r.get('id');part=r.get('partition');p=s['partitions'].setdefault(str(part),{'records':0,'missing_fields':C.Counter(),'null_fields':C.Counter(),'empty_fields':C.Counter(),'invalid_field_types':C.Counter(),'slot_types':C.Counter(),'slot_count_distribution':C.Counter(),'exact_restore':0,'whitespace_insensitive_restore':0,'parse_errors':0})
                p['records']+=1
                for k in FIELDS:
                    if k not in r:p['missing_fields'][k]+=1
                    elif r[k] is None:p['null_fields'][k]+=1
                    elif r[k] in ('',[],{}):p['empty_fields'][k]+=1
                    s['field_types'].setdefault(k,C.Counter())[type(r[k]).__name__ if k in r else 'MISSING']+=1
                    if k in r and not isinstance(r[k],list if k in ('slot_method','judgments') else str):p['invalid_field_types'][k]+=1
                if r.get('locale')!=loc:s['unexpected_locale_rows']+=1;anomalies.append({'locale':loc,'id':rid,'line':ln,'kind':'locale_mismatch','detail':js(r.get('locale'))})
                if part not in ('train','dev','test'):s['invalid_partition_rows']+=1;anomalies.append({'locale':loc,'id':rid,'line':ln,'kind':'invalid_partition','detail':js(part)})
                slots,restored,errs=parse_annotation(r.get('annot_utt'));utt=r.get('utt')
                ex=isinstance(utt,str) and restored==utt;ws=isinstance(utt,str) and compact(restored)==compact(utt)
                s['parse_error_rows']+=bool(errs);s['exact_restore_rows']+=ex;s['whitespace_insensitive_restore_rows']+=ws
                p['exact_restore']+=ex;p['whitespace_insensitive_restore']+=ws;p['parse_errors']+=bool(errs)
                for k in ('slot_types',):s[k].update(signature(slots));p[k].update(signature(slots))
                s['slot_count_distribution'][len(slots)]+=1;p['slot_count_distribution'][len(slots)]+=1
                methods=r.get('slot_method',[])
                if not isinstance(methods,list):methods=[];anomalies.append({'locale':loc,'id':rid,'line':ln,'kind':'invalid_slot_method_structure','detail':''})
                ms=set()
                for m in methods:
                    if not isinstance(m,dict):anomalies.append({'locale':loc,'id':rid,'line':ln,'kind':'invalid_method_item','detail':js(m)});continue
                    s['method_item_keysets'][js(sorted(m))]+=1;s['method_values'][str(m.get('method'))]+=1;s['method_slot_types'][str(m.get('slot'))]+=1;ms.add(m.get('slot'))
                    if m.get('method') not in ('translation','localization','unchanged'):anomalies.append({'locale':loc,'id':rid,'line':ln,'kind':'extended_method_value','detail':js(m)})
                method_eq=ms==set(signature(slots)) if 'slot_method' in r else None
                if method_eq is False:s['method_annotation_type_set_mismatch_rows']+=1;anomalies.append({'locale':loc,'id':rid,'line':ln,'kind':'method_annotation_type_set_difference','detail':js({'method_types':sorted(ms),'annotation_types':sorted(signature(slots))})})
                judgments=r.get('judgments',[])
                if 'judgments' in r:
                    if not isinstance(judgments,list):anomalies.append({'locale':loc,'id':rid,'line':ln,'kind':'invalid_judgments_structure','detail':''});judgments=[]
                    s['judgment_length_distribution'][len(judgments)]+=1
                    if len(judgments)!=3:anomalies.append({'locale':loc,'id':rid,'line':ln,'kind':'judgment_count_not_three','detail':str(len(judgments))})
                    per=C.defaultdict(set)
                    for j in judgments:
                        if not isinstance(j,dict):anomalies.append({'locale':loc,'id':rid,'line':ln,'kind':'invalid_judgment_item','detail':js(j)});continue
                        s['judgment_keysets'][js(sorted(j))]+=1
                        for k in set(JDOM)|{'worker_id','language_identification'}:
                            if k not in j:s['judgment_missing_keys'][k]+=1
                        for k,v in j.items():
                            s['judgment_values'].setdefault(k,C.Counter())[js(v)]+=1
                            s['judgment_field_types'].setdefault(k,C.Counter())[type(v).__name__]+=1
                            per[k].add(js(v))
                            invalid=(k in JDOM and (type(v) is not int or v not in JDOM[k])) or (k=='language_identification' and (not isinstance(v,str) or not set(v.split('|'))<={'target','english','other'}))
                            if invalid:s['judgment_out_of_domain'][k]+=1
                    for k,vs in per.items():
                        if k!='worker_id' and len(vs)>1:s['judgment_disagreement_rows'][k]+=1
                if errs or not ws:anomalies.append({'locale':loc,'id':rid,'line':ln,'kind':'annotation_parse_or_restore','detail':js({'errors':errs,'whitespace_restore':ws})})
                sw.writerow(dict(zip(slotfields,[loc,part,rid,name,ln,utt,r.get('annot_utt'),restored,ex,ws,js(errs),len(slots),js(slots),js(r.get('slot_method')),'' if method_eq is None else method_eq])))
                if not isinstance(rid,str) or not rid:
                    anomalies.append({'locale':loc,'id':rid,'line':ln,'kind':'invalid_id','detail':'excluded from joins'});continue
                canonical=js(r);payload=js({k:v for k,v in r.items() if k not in ('id','worker_id','judgments','slot_method')})
                con.execute('INSERT INTO rec VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',(loc,rid,str(part),r.get('intent'),r.get('scenario'),utt,r.get('annot_utt'),js(slots),js(methods),ln,hashlib.sha256(canonical.encode()).hexdigest(),hashlib.sha256(payload.encode()).hexdigest()))
            con.commit()
            manifest['members'].append({'path':name,'sha256':h.hexdigest(),'size_bytes':t.getmember(name).size,'physical_lines':s['physical_lines']})
            s['distinct_ids']=con.execute('SELECT COUNT(DISTINCT id) FROM rec WHERE loc=?',(loc,)).fetchone()[0]
            s['duplicate_id_groups']=con.execute('SELECT COUNT(*) FROM (SELECT id FROM rec WHERE loc=? GROUP BY id HAVING COUNT(*)>1)',(loc,)).fetchone()[0]
            for part,p in s['partitions'].items():
                p['distinct_ids']=con.execute('SELECT COUNT(DISTINCT id) FROM rec WHERE loc=? AND part=?',(loc,part)).fetchone()[0]
                for col,kind in [('id','id'),('rawhash','canonical_full_record'),('payloadhash','content_excluding_id_and_worker_metadata'),('utt','exact_utterance')]:
                    ns=[x[0] for x in con.execute(f'SELECT COUNT(*) FROM rec WHERE loc=? AND part=? GROUP BY {col} HAVING COUNT(*)>1',(loc,part))]
                    p[kind+'_duplicate_groups']=len(ns);p[kind+'_duplicate_excess_records']=sum(n-1 for n in ns)
            print(f'{loc}: {s["valid_json_objects"]} rows; {s["distinct_ids"]} IDs; {s["parse_error_rows"]} parser errors',flush=True)
        sf.close()
        duplicates=[]
        for loc in LOCALES:
            for col,kind in [('id','id'),('rawhash','canonical_full_record'),('payloadhash','content_excluding_id_and_worker_metadata'),('utt','exact_utterance')]:
                groups=0;excess=0;cross=0
                for key,n in con.execute(f'SELECT {col},COUNT(*) FROM rec WHERE loc=? GROUP BY {col} HAVING COUNT(*)>1',(loc,)):
                    rows=con.execute(f'SELECT id,part,line FROM rec WHERE loc=? AND {col}=? ORDER BY line',(loc,key)).fetchall()
                    groups+=1;excess+=n-1;cross+=len(set(x[1] for x in rows))>1
                    duplicates.append({'locale':loc,'kind':kind,'key':key,'records':n,'ids_json':js([x[0] for x in rows]),'partitions_json':js([x[1] for x in rows]),'source_lines_json':js([x[2] for x in rows])})
                stats[loc][kind+'_duplicates']={'groups':groups,'excess_records':excess,'cross_partition_groups':cross}
        write_csv(out/'duplicate_records.csv',['locale','kind','key','records','ids_json','partitions_json','source_lines_json'],duplicates)
        intersections=[]
        for loc in LOCALES:
            for a,b in itertools.combinations(sorted(stats[loc]['partitions']),2):
                ids=[r[0] for r in con.execute('SELECT id FROM rec WHERE loc=? AND part=? INTERSECT SELECT id FROM rec WHERE loc=? AND part=?',(loc,a,loc,b))]
                intersections.append({'locale':loc,'partition_a':a,'partition_b':b,'intersection_count':len(ids),'ids_json':js(ids)})
        write_csv(out/'partition_intersections.csv',['locale','partition_a','partition_b','intersection_count','ids_json'],intersections)
        summary['partition_intersections']=intersections
        align_fields=['id']+[f'{l}_{k}' for l in LOCALES for k in ('count','partition','intent','scenario','source_lines')]+['three_way_unique','partition_consistent','intent_consistent','scenario_consistent','slot_type_multiset_consistent']
        diff_fields=['id','partition_a','partition_b','locale_a','locale_b','intent_a','intent_b','slot_type_set_equal','slot_type_multiset_equal','slot_count_equal','slot_value_multiset_equal','slots_a_json','slots_b_json','methods_a_json','methods_b_json','surface_observation','semantic_status','localization_metadata_present']
        af=(out/'language_alignment.csv').open('w',newline='',encoding='utf-8');aw=csv.DictWriter(af,align_fields);aw.writeheader()
        df=(out/'localization_differences.csv').open('w',newline='',encoding='utf-8');dw=csv.DictWriter(df,diff_fields);dw.writeheader()
        pair_counts={f'{a}__{b}':C.Counter() for a,b in itertools.combinations(LOCALES,2)};trip=C.Counter()
        ids=[r[0] for r in con.execute('SELECT DISTINCT id FROM rec ORDER BY CAST(id AS INTEGER),id')]
        for rid in ids:
            by={l:[] for l in LOCALES}
            for row in con.execute('SELECT loc,part,intent,scenario,slots,methods,line FROM rec WHERE id=? ORDER BY loc,line',(rid,)):
                loc,part,intent,scenario,slots,methods,line=row;by[loc].append({'partition':part,'intent':intent,'scenario':scenario,'slots':json.loads(slots),'methods':json.loads(methods),'line':line})
            rr={'id':rid}
            for l in LOCALES:
                rr[f'{l}_count']=len(by[l]);rr[f'{l}_source_lines']=js([x['line'] for x in by[l]])
                for k in ('partition','intent','scenario'):rr[f'{l}_{k}']=js([x[k] for x in by[l]])
            common=all(by[l] for l in LOCALES);unique=all(len(by[l])==1 for l in LOCALES);trip['common_ids']+=common;trip['unique_triples']+=unique;rr['three_way_unique']=unique
            for k in ('partition','intent','scenario'):
                eq=len({x[k] for rows in by.values() for x in rows})==1 if common else None
                rr[k+'_consistent']='' if eq is None else eq
                if common:trip[k+'_mismatch_ids']+=not eq
            eq=unique and len({js(signature(by[l][0]['slots'])) for l in LOCALES})==1
            rr['slot_type_multiset_consistent']=eq if unique else '';trip['slot_type_multiset_mismatch_ids']+=unique and not eq
            aw.writerow(rr)
            for a,b in itertools.combinations(LOCALES,2):
                c=pair_counts[f'{a}__{b}'];c['common_ids']+=bool(by[a] and by[b]);c['a_only_ids']+=bool(by[a] and not by[b]);c['b_only_ids']+=bool(by[b] and not by[a])
                if len(by[a])!=1 or len(by[b])!=1:continue
                x,y=by[a][0],by[b][0];f=pair_flags(x,y);c['unique_pairs']+=1
                for k,v in f.items():c[k]+=v
                c['intent_mismatch']+=x['intent']!=y['intent'];c['partition_mismatch']+=x['partition']!=y['partition']
                localized=any(m.get('method') in ('localization','unchanged_localization') for m in x['methods']+y['methods']);c['localization_metadata_present']+=localized
                dw.writerow({'id':rid,'partition_a':x['partition'],'partition_b':y['partition'],'locale_a':a,'locale_b':b,'intent_a':x['intent'],'intent_b':y['intent'],**f,'slots_a_json':js(x['slots']),'slots_b_json':js(y['slots']),'methods_a_json':js(x['methods']),'methods_b_json':js(y['methods']),'surface_observation':'same_slot_strings' if f['slot_value_multiset_equal'] else 'string_difference','semantic_status':'not_adjudicated','localization_metadata_present':localized})
        af.close();df.close();con.close()
        summary['alignment']={'triple':trip,'pairs':pair_counts,'union_ids':len(ids)}
    write_csv(out/'data_anomalies.csv',['locale','id','line','kind','detail'],anomalies)
    summary['anomaly_counts']=C.Counter(x['kind'] for x in anomalies)
    manifest['post_run_sha256']=sha(source);manifest['source_unchanged']=manifest['post_run_sha256']==before
    if not manifest['source_unchanged']:raise RuntimeError('Source changed during audit')
    dump(out/'source_manifest.json',manifest);dump(out/'dataset_quality_summary.json',summary)
    dump(out/'run_metadata.json',{'command_argv':sys.argv,'cwd':str(Path.cwd()),'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'dependencies':'Python standard library only; no third-party packages','utc_completed':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started,'script_sha256':sha(__file__),'model_calls':0,'training_runs':0,'new_human_labels':0})
    print(js(summary['alignment']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--scratch');a=p.parse_args();audit(a.source,a.out,a.scratch)
