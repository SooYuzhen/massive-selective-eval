"""Static release-tree audit. No network, credentials or model loading."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    findings=[];scanned=[]
    patterns={'personal_home':re.compile('/'+'Users/[^/\\s]+/|/'+'home/[^/\\s]+/'),'credential':re.compile(r'\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16})\b'),'private_key':re.compile('-----BEGIN '+'(?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),'email':re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')}
    banned={'.safetensors','.joblib','.parquet','.npz','.pt','.pth','.gguf','.bin','.pem','.key','.env'}
    for p in sorted(ROOT.rglob('*')):
        rel=str(p.relative_to(ROOT))
        if '__pycache__' in p.parts or any(x.endswith('.egg-info') for x in p.parts):continue
        if p.is_symlink():findings.append({'path':rel,'kind':'symlink'})
        if not p.is_file() or rel in ['audit/security_scan.json','audit/file_manifest.json']:continue
        if p.suffix=='.zip':continue
        if p.suffix in banned or p.name.startswith('.env') or p.name=='.DS_Store':findings.append({'path':rel,'kind':'disallowed_file'})
        if p.stat().st_size>2_000_000:findings.append({'path':rel,'kind':'oversized_file'})
        data=p.read_bytes()
        try:text=data.decode('utf-8')
        except UnicodeError:findings.append({'path':rel,'kind':'unexpected_binary'});continue
        for kind,pattern in patterns.items():
            for match in pattern.finditer(text):findings.append({'path':rel,'kind':kind,'line':text[:match.start()].count('\n')+1})
        scanned.append({'path':rel,'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    result={'status':'PASS' if not findings else 'FAIL','scanned_files':len(scanned),'findings':findings,'limits':'Pattern and file-type audit only; not an exhaustive legal or secret-detection guarantee'}
    (ROOT/'audit/security_scan.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return not findings
if __name__=='__main__':raise SystemExit(0 if main() else 1)
