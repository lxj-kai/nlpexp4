with open(r'D:\code\nlpexp4\report_latex\figures\system_architecture.html', 'rb') as f:
    raw = f.read()

print(f'Size: {len(raw)}')
print(f'First 10 hex: {raw[:10].hex()}')
null_count = raw.count(b'\x00')
print(f'Null bytes: {null_count}')
has_bom = raw[:3] == b'\xef\xbb\xbf'
print(f'UTF-8 BOM: {has_bom}')

# Check for \r\n
crlf = raw.count(b'\r\n')
lf_only = raw.count(b'\n') - crlf
print(f'CRLF: {crlf}, LF only: {lf_only}')
