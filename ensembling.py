import rdkit
from rdkit.Chem import MolFromInchi
from rdkit import RDLogger

from tqdm import tqdm
import pandas as pd
from pathlib import Path

RDLogger.DisableLog('rdApp.*')


def check_inchi(inchi):
    mol = MolFromInchi(inchi)
    
    if mol is not None:
        return True
    else:
        return False


weak = pd.read_csv("")["InChI"].values
strong = pd.read_csv("")["InChI"].values
sub = pd.read_csv("./sample_submission.csv") 
ids = pd.read_csv("")["image_id"].values
norm_path = Path("")

N = norm_path.read_text().count('\n') if norm_path.exists() else 0
print(N, 'number of predictions already processed')

write_mode = 'w' if N == 0 else 'a'
w = open(str(norm_path), write_mode, buffering=1)
w.write(f'{ids[N]},"{strong[N]}"\n')

true_strong, true_weak, fake = 0, 0, 0
values = []

for i in tqdm(range(N+1, len(weak))):

    image_id = ids[i]
    o = weak[i]
    n = strong[i]

    if check_inchi(n):
        s = n
        true_strong += 1

    else:
        if check_inchi(o):
            s = o
            true_weak +=1
        else:
            s = n
            fake += 1

    if i % 100000 == 0:
        print(true_strong, true_weak, fake)
        
    w.write(f'{image_id},"{s}"\n')
