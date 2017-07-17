

# backward differences


def backward_dcdx(m, s, x):
    idx = m.x.index(x)
    dx = m.x[idx] - m.x[idx - 1]
    i = m.s_to_id[s]
    j = idx
    jb = idx - 1
    return (m.C[i, j] - m.C[i, jb]) / dx


def backward_dcdx2(m, s, x):
    idx = m.x.index(x)
    dx = m.x[idx] - m.x[idx - 1]
    i = m.s_to_id[s]
    j = idx
    jb = idx - 1
    jbb = idx - 2
    return (m.C[i, j] - 2 * m.C[i, jb] + m.C[i, jbb])/dx**2