import pandas as pd
from pathlib import Path
import sklearn
from glob import iglob

pdList = []
for year in range(2007,2020):
    newdf = None
    csv_files = Path.cwd().glob('Complete_%s\\*.csv' % str(year))
    for filename in csv_files:
        print(filename)
        newdf = pd.read_csv(filename)
        print(newdf)
    pdList.append(newdf)

outdf = pd.concat(pdList)


outdf.to_csv('out.csv', index=True)