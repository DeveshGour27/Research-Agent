import os
files_to_fix = [
    'app/llm/provider.py',
    'app/llm/gateway.py',
    'app/llm/factory.py',
    'app/llm/base.py'
]
for p in files_to_fix:
    if os.path.exists(p):
        c = open(p, encoding='utf-8').read()
        if '"\""' in c:
            c = c.replace('"\""', '"""')
            open(p, 'w', encoding='utf-8').write(c)
