
adj = [(0,1),(0,-1),(1,-1),(1,0),(-1,1),(-1,0)]

def dfs(g,player_id,size):
    visited = set()
    p = {}
    for u in g:
        if u[player_id - 1] != 0:
            continue
        if u not in visited:
            p[(u[0],u[1])] = None
            if dfs_visit(g,u,visited,p,size,player_id):
                return True
    return False

def dfs_visit(g,u,visited,p,size,player_id):
    visited.add((u))
    for dir in adj:
        v = (u[0]+dir[0],u[1]+dir[1])
        if v not in g:
            continue
        if v[player_id - 1] == size - 1:
            p[v] = u
            return True
        if v not in visited:
            p[v] = u
            if dfs_visit(g,v,visited,p,size,player_id):
                return True
    return False