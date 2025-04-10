
adj = [(0,1), (0,-1), (1,-1), (1,0), (-1,1), (-1,0)]

def dfs(g, player_id, size):
    visited = set()
    p = {}
    for u in g:
        # Verificar si el nodo pertenece al borde inicial según el jugador
        if player_id == 1 and u[1] != 0:
            continue
        elif player_id == 2 and u[0] != 0:
            continue

        if u not in visited:
            p[(u[0], u[1])] = None
            result = dfs_visit(g, u, visited, p, size, player_id)
            if result is not None:
                # Retornar los extremos del camino ganador
                start_node = u
                end_node = result
                return True, (start_node, end_node)
    return False, (None, None)

def dfs_visit(g, u, visited, p, size, player_id):
    visited.add(u)
    for dir in adj:
        v = (u[0] + dir[0], u[1] + dir[1])
        if v not in g:
            continue
        # Verificar si se llega al borde opuesto según el jugador
        if player_id == 1 and v[1] == size - 1:
            return v  # Nodo final ganador
        elif player_id == 2 and v[0] == size - 1:
            return v  # Nodo final ganador

        if v not in visited:
            p[v] = u
            result = dfs_visit(g, v, visited, p, size, player_id)
            if result is not None:
                return result
    return None