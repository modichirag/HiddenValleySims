import numpy as np

def readhead(path):
    shape = None
    with open(path +'attr-v2') as  f:
        for line in f.readlines():
            if 'ndarray.shape' in line:
                shape = tuple(int(i) for i in line.split('[')[1].split()[:-1])
    with open(path +'header') as  f:
        for line in f.readlines():
            if 'DTYPE' in line: dtype = line.split()[-1]
            if 'NFILE' in line: nf = int(line.split()[-1])
            if 'NMEMB' in line:
                if shape is None: shape = tuple([-1, int(line.split()[-1])])
    return dtype, nf, shape


def readbigfile(path):
    dtype, nf, shape = readhead(path)
    data = []
    for i in range(nf): data.append(np.fromfile(path + '%06d'%i, dtype=dtype))
    data = np.concatenate(data)
    data = np.reshape(data, shape)
    return data


