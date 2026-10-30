"""Stream existing local records; never download or extract tar members to disk."""
import json,tarfile
from pathlib import Path
from .raw_audit import verify_archive

def iter_records(source,locales=('en-US','zh-CN','ja-JP')):
    source=Path(source)
    if source.is_dir():
        files=sorted(source.glob('*.jsonl')) or sorted((source/'data').glob('*.jsonl'))
        if not files:raise ValueError('No JSONL files in the directory or its data subdirectory')
        for file in files:
            yield from iter_records(file,locales)
    elif source.suffix=='.parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(source).iter_batches(batch_size=1024):
            for r in batch.to_pylist():
                if r.get('locale') in locales:yield r
    elif source.suffix in ('.jsonl','.ndjson'):
        with source.open(encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r=json.loads(line)
                    if r.get('locale') in locales:yield r
    else:
        with tarfile.open(source,'r:*') as archive:
            verify_archive(archive)
            for locale in locales:
                f=archive.extractfile(f'1.1/data/{locale}.jsonl')
                for line in f:
                    if line.strip():yield json.loads(line)

def load_to_jsonl(source,output,locales,partition=None,membership=None):
    import csv
    allowed=None
    if membership:
        with Path(membership).open(newline='') as f:
            allowed={(r['locale'],r['id']) for r in csv.DictReader(f) if r['retain']=='True'}
    counts={};seen=set()
    with Path(output).open('x',encoding='utf-8') as f:
        for r in iter_records(source,locales):
            if partition and r.get('partition')!=partition:continue
            key=(r['locale'],str(r['id']))
            if allowed is not None and key not in allowed:continue
            if key in seen:raise ValueError(f'Duplicate locale/source ID: {key}')
            seen.add(key)
            if not isinstance(r.get('utt'),str):raise ValueError('Missing utterance')
            f.write(json.dumps(r,ensure_ascii=False)+'\n');counts[r['locale']]=counts.get(r['locale'],0)+1
    return {'records':len(seen),'per_locale':counts,'version_note':'Tar identity checked as original 1.1; standalone JSONL/Parquet requires caller provenance.'}
