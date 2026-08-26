import os
path = "app/evaluation/benchmark_dataset.py"
c = open(path, encoding="utf-8").read()
c = c.replace('encoding="utf-8"', 'encoding="utf-8-sig"')
open(path, "w", encoding="utf-8").write(c)
