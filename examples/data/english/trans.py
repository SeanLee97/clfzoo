# -*- coding: utf-8- -*-

def process(fin, fout):
    outformat = '{}\t{}\n'
    with open(fin, 'r') as rf, open(fout, 'w') as wf:
        for line in rf:
            line = line.strip()
            arr = line.split()
            if len(arr) == 0:
                continue

            wf.writelines(outformat.format(arr[0], ' '.join(arr[1:])))

process('TREC.train.txt', 'TREC.train')
process('TREC.test.txt', 'TREC.test')
