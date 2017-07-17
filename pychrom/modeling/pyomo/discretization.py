
# backward differences


def backward_dcdx(m, s, t, x, x_list):
    idx = x_list.index(x)
    dx = x_list[idx] - x_list[idx - 1]
    return (m.C[s, t, x_list[idx]] - m.C[s, t, x_list[idx - 1]]) / dx


def backward_dcdx2(m, s, t, x, x_list):
    idx = x_list.index(x)
    dx = x_list[idx] - x_list[idx-1]
    if idx-2 > 0:
        return (m.C[s, t, x_list[idx]] - 2 * m.C[s, t, x_list[idx-1]] + m.C[s, t, x_list[idx-2]])/dx**2
    else:  # silently ignores first point
        return 0.0


# forward differences


def forward_dcdx(m, s, t, x, x_list):
    idx = x_list.index(x)
    dx = x_list[idx+1] - x_list[idx]
    return (m.C[s, t, x_list[idx+1]] - m.C[s, t, x_list[idx]]) / dx


def forward_dcdx2(m, s, t, x, x_list):
    idx = x_list.index(x)
    dx = x_list[idx+1] - x_list[idx]
    if idx + 2 < len(x_list)-1:
        return (m.C[s, t, x_list[idx+2]] - 2 * m.C[s, t, x_list[idx+1]] + m.C[s, t, x_list[idx]])/dx**2
    else:  # silently ignores last point
        return 0.0
