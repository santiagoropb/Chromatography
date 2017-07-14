
# backward differences


def backward_dcdx(v,s,t,x,x_list):
    idx = x_list.index(x)
    dx = x_list[idx] - x_list[idx - 1]
    return (v[s, t, x_list[idx]] - v[s, t, x_list[idx - 1]]) / dx


def backward_dcdx2(v,s,t,x, x_list):
    idx = x_list.index(x)
    dx = x_list[idx] - x_list[idx-1]
    return (v[s,t, x_list[idx]] - 2 * v[s, t, x_list[idx-1]] + v[s, t, x_list[idx-2]])/dx**2


# forward differences


def forward_dcdx(v,s,t,x,x_list):
    idx = x_list.index(x)
    dx = x_list[idx+1] - x_list[idx]
    return (v[s, t, x_list[idx+1]] - v[s, t, x_list[idx]]) / dx


def forward_dcdx2(v,s,t,x,x_list):
    idx = x_list.index(x)
    dx = x_list[idx+1] - x_list[idx]
    return (v[s, t, x_list[idx+2]] - 2 * v[s, t, x_list[idx+1]] + v[s, t, x_list[idx]])/dx**2
