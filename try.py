from tqdm import tqdm

with tqdm(total = 10000) as pbar:
    for i in range(10000):
        pbar.update(1)
