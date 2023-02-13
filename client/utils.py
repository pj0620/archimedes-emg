def write_samples_to_disk(data: list[list[int]], fname:str):
    f = open(f'data/{fname}.csv', 'w')
    for sample in data:
        f.write(','.join(str(s) for s in sample) + '\n')
    f.close()
