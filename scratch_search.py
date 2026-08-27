import os

search_terms = ['ToolCalledEvent', 'MemoryRetrievedEvent', 'Model request completed', 'Model request failed']
for root, dirs, files in os.walk('app'):
    if '__pycache__' in root:
        continue
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for term in search_terms:
                    if term in content:
                        print(f"Found {term} in {filepath}")
