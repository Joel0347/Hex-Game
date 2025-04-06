from numpy import random
from player_def import Player
from board import HexBoard
from collections import deque
from utils import dfs

class AI_Player(Player):
    def __init__(self, player_id):
        super().__init__(player_id)
        
    def play(self, board: HexBoard) -> tuple[int, int]:
        """Calcula y devuelve la jugada de la PC"""
        
        if all(all(element == 0 for element in row) for row in board.board):
            return (board.size // 2, board.size // 2)
        
        # Primero verifica si hay movimientos ganadores inmediatos
        winning_move = self.look_for_win_next_round(board, self.player_id)
        if winning_move:
            return winning_move
        
        # Luego verifica si el oponente puede ganar en el siguiente movimiento
        blocking_move = self.look_for_win_next_round(board, 3 - self.player_id)
        if blocking_move:
            return blocking_move
        
        # Decide la estrategia basada en la fase del juego
        empty_cells = sum(row.count(0) for row in board.board)
        game_phase = empty_cells / (board.size ** 2)
        
        if game_phase > 0.85:
            return self.monte_carlo_method(board, self.player_id)
        else:
            depth = self.calculate_depth_limit(board)
            return self.minimax(board, depth)[0]

    
    def minimax(
        self, board: HexBoard, depth: int, level_parity=True, 
        alpha=-float('inf'), betha=float('inf')
        ) -> tuple[tuple[int, int], int]:
        """
        Algoritmo Minimax para buscar en profundidad 
        limitada la mejor jugada
        """
        
        id = self.player_id if level_parity else 3 - self.player_id
        
        if (not depth):
            return (), self.heuristic_critical_moves(board, id)
        
        if board.check_connection(3 - self.player_id):
            return (), 0
        elif board.check_connection(self.player_id):
            return (), 1
        
        possible_moves = board.get_possible_moves()
        next_move = possible_moves[0]
        
        for row, col in possible_moves:
            AI_Player.place_piece(board, row, col, id)
            
            if level_parity:
                move_eval = self.minimax(
                    board, depth - 1, not level_parity, alpha=alpha
                )[1] 
            else:
                move_eval = self.minimax(
                    board, depth - 1, not level_parity, betha=betha
                )[1]

            AI_Player.remove_piece(board, row, col, id)
            
            if level_parity and alpha < move_eval:
                alpha = move_eval
                next_move = (row, col)
            elif (not level_parity) and betha > move_eval:
                betha = move_eval
                next_move = (row, col)
            
            if betha <= alpha:
                result = betha if level_parity else alpha
                return (), result
            
        result = alpha if level_parity else betha
        
        return next_move, result
    
    def calculate_depth_limit(self, board: HexBoard) -> int:
        """Calcula el límite de profundidad a explorar"""
        
        possible_moves = board.get_possible_moves()
        times_AI_already_played = sum(
            [row.count(self.player_id) for row in board.board]
        )
        times_opponent_already_played = sum(
            [row.count(3 - self.player_id) for row in board.board]
        )
        total_played = times_AI_already_played + times_opponent_already_played
        max_to_play = board.size ** 2
        
        if len(possible_moves) < 10:
            return float('inf')
        else:
            return 3 + (total_played // max_to_play) * 2

    @staticmethod
    def simulate(board: HexBoard, player_id: int, player_on_turn: int) -> int:
        """Simula cada jugada de manera aleatoria hasta que gane alguien"""
        
        if board.check_connection(player_id):
            result = 1
        elif board.check_connection(3 - player_id):
            result = 0
        else:
            possible_moves = board.get_possible_moves()
            row, col = possible_moves[random.choice(len(possible_moves))]
            AI_Player.place_piece(board, row, col, player_on_turn)
            result = AI_Player.simulate(board, player_id, 3 - player_on_turn)
            AI_Player.remove_piece(board, row, col, player_on_turn)

        return result

    @staticmethod
    def monte_carlo_method(board: HexBoard, player_id: int, simulations=2000) -> tuple[int, int]:
        """
        Método de monte carlo para simular las jugadas hasta que alguien gane y 
        contar la cantidad de victorias en cada jugada realizada por simulación, 
        así se escoge la que más victorias haya producido
        """
        
        possible_moves = board.get_possible_moves()
        move_win_counts = {move: 0 for move in possible_moves}

        while simulations > 0:
            move = possible_moves[random.randint(0, len(possible_moves))]
            AI_Player.place_piece(board, move[0], move[1], player_id)
            move_win_counts[move] += AI_Player.simulate(board, player_id, 3 - player_id)
            AI_Player.remove_piece(board, move[0], move[1], player_id)
            
            simulations -= 1
        
        return max(move_win_counts, key=move_win_counts.get)

    @staticmethod
    def heuristic_critical_moves(board: HexBoard, player_id: int) -> int:
        """
        Heurística que evalúa la menor cantidad de movimientos para ganar y 
        devuelve el inverso para asegurar que las jugadas más cercanas a la 
        victoria tengan mejor puntuación
        """
        
        score = 1 / AI_Player.minimum_moves_to_win(board, player_id)
        return score

    @staticmethod
    def minimum_moves_to_win(board: HexBoard, player_id: int) -> int:
        """Calcula la menor cantidad de jugadas a la victoria con BFS"""
        
        # Determinar los bordes inicial y final según el jugador
        size = len(board.board)
        if player_id == 1:
            start = [(i, 0) for i in range(size) if board.board[i][0] != 3 - player_id]
            end = [(i, size - 1) for i in range(size) if board.board[i][size - 1] != 3 - player_id]
        else:
            start = [(0, i) for i in range(size) if board.board[0][i] != 3 - player_id]
            end = [(size - 1, i) for i in range(size) if board.board[size - 1][i] != 3 - player_id]

        # Calcular movimientos mínimos necesarios
        return AI_Player.bfs(board, player_id, start, end)
    
    @staticmethod
    def bfs(board: HexBoard, player_id: int, start: list[tuple[int, int]], end: list[tuple[int, int]]) -> int:
        """
        Algoritmo BFS para calcular la menor cantidad de movimientos para ganar, penalizando caminos
        que tengan que pasar por casillas vacías, para buscar caminos ya hechos anteriormente 
        hasta cierto punto
        """
        
        # Implementación de búsqueda de camino más corto (BFS)
        shortest_path = float('inf')
        for node in start:
            queue = deque([(node, 0)])  # Nodo actual y profundidad
            visited = set()
            size = board.size

            while queue:
                (x, y), depth = queue.popleft()
                
                if (x, y) in visited:
                    continue
                
                visited.add((x, y))
                
                # Si se alcanza el objetivo
                if (x, y) in end:
                    shortest_path = min(shortest_path, depth)
                    break
                
                # Explorar vecinos válidos
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited:
                        if board.board[nx][ny] == player_id:
                            queue.append(((nx, ny), depth))
                        elif board.board[nx][ny] == 0:
                            queue.append(((nx, ny), depth + 1))
        
        return shortest_path
    
    @staticmethod
    def look_for_win_next_round(board: HexBoard, player_id: int) -> tuple[int, int]:
        """Busca alguna opción con 100% de probabilidades de victoria en próxima ronda"""
        
        possible_moves = board.get_possible_moves()
        inmediate_victory = AI_Player.find_inmediate_win(board, possible_moves, player_id)
        
        if inmediate_victory:
            return inmediate_victory
        
        for row, col in possible_moves:
            AI_Player.place_piece(board, row, col, player_id)
            if AI_Player.more_than_one_chance_for_winning(board, player_id):
                AI_Player.remove_piece(board, row, col, player_id)
                return row, col
            AI_Player.remove_piece(board, row, col, player_id)
            
        return ()
    
    @staticmethod
    def find_inmediate_win(
        board: HexBoard, possible_moves: list[tuple[int, int]], 
        player_id: int
        ) -> tuple[int, int]:
        """Busca alguna victoria en la jugada actual"""
        
        for row, col in possible_moves:
            AI_Player.place_piece(board, row, col, player_id)
            if AI_Player.check_connection(board, player_id):
                AI_Player.remove_piece(board, row, col, player_id)
                return row, col
            AI_Player.remove_piece(board, row, col, player_id)
            
        return ()
    
    @staticmethod
    def more_than_one_chance_for_winning(board: HexBoard, player_id: int, check_next_move=True) -> bool:
        """
        Revisa si el movimiento realizado anteriormente permite 100% de 
        probabilidades de victoria en la próxima ronda
        """
        
        size = board.size
        win_area = [
            (i, j) for i in range(size) for j in [0, size - 1]
            ] if player_id == 1 else [
                (i, j) for j in range(size) for i in [0, size - 1]
                ]
        wins_count = 0
        for row, col in win_area:
            if board.board[row][col] != 0:
                continue
            AI_Player.place_piece(board, row, col, player_id)
            if (AI_Player.check_connection(board, player_id)):
                wins_count += 1
            AI_Player.remove_piece(board, row, col, player_id)
            
            if wins_count > 1:
                return True
            
        if check_next_move:
            return AI_Player.chances_for_winning_after_next_round(board, player_id)
        else:
            return False
    
    @staticmethod
    def build_reduced_board(size: int, positions: dict[int: list[tuple[int, int]]]) -> HexBoard:
        """Construye un tablero reducido a partir del tablero real"""
        
        new_board = HexBoard(size)
        for player in positions:
            for row, col in positions[player]:
                AI_Player.place_piece(new_board, row, col, player)
                
        return new_board
    
    @staticmethod
    def chances_for_winning_after_next_round(board: HexBoard, player_id: int) -> bool:
        """Busca posibilidades de victoria en las próximas 2 jugadas"""
        
        player_pos = [
            (i, j) for i, row in enumerate(board.board) 
            for j, value in enumerate(row) if value == player_id
            ]
        
        opponent_pos = [
            (i, j) for i, row in enumerate(board.board) 
            for j, value in enumerate(row) if value == 3 - player_id
            ]
        
        # eliminamos primero la primera fila y primera columna
        player_pos_reduced = [(pos[0] - 1, pos[1] - 1) for pos in player_pos if pos[0] > 0 and pos[1] > 0]
        opponent_pos_reduced = [(pos[0] - 1, pos[1] - 1) for pos in opponent_pos if pos[0] > 0 and pos[1] > 0]
        positions = {player_id: player_pos_reduced, (3-player_id): opponent_pos_reduced}
        board_without_up_left = AI_Player.build_reduced_board(board.size - 1, positions)
        
        # eliminamos ahora la primera fila y la última columna
        player_pos_reduced = [(pos[0] - 1, pos[1]) for pos in player_pos if pos[0] > 0 and pos[1] < board.size - 1]
        opponent_pos_reduced = [(pos[0] - 1, pos[1]) for pos in opponent_pos if pos[0] > 0 and pos[1] < board.size - 1]
        positions = {player_id: player_pos_reduced, (3-player_id): opponent_pos_reduced}
        board_without_up_right = AI_Player.build_reduced_board(board.size - 1, positions)
        
        # eliminamos primero la última fila y primera columna
        player_pos_reduced = [(pos[0], pos[1] - 1) for pos in player_pos if pos[0] < board.size - 1 and pos[1] > 0]
        opponent_pos_reduced = [(pos[0], pos[1] - 1) for pos in opponent_pos if pos[0] < board.size - 1 and pos[1] > 0]
        positions = {player_id: player_pos_reduced, (3-player_id): opponent_pos_reduced}
        board_without_down_left = AI_Player.build_reduced_board(board.size - 1, positions)
        
        # eliminamos ahora la última fila y la última columna
        player_pos_reduced = [pos for pos in player_pos if pos[0] < board.size - 1 and pos[1] < board.size - 1]
        opponent_pos_reduced = [pos for pos in opponent_pos if pos[0] < board.size - 1 and pos[1] < board.size - 1]
        positions = {player_id: player_pos_reduced, (3-player_id): opponent_pos_reduced}
        board_without_down_right = AI_Player.build_reduced_board(board.size - 1, positions)
        
        return (
            AI_Player.more_than_one_chance_for_winning(board_without_up_left, player_id, False) and
            AI_Player.more_than_one_chance_for_winning(board_without_up_right, player_id, False)
        ) or (
            AI_Player.more_than_one_chance_for_winning(board_without_down_left, player_id, False) and
            AI_Player.more_than_one_chance_for_winning(board_without_down_right, player_id, False)
        )
        
        
    @staticmethod
    def place_piece(board: HexBoard, row: int, col: int, player_id: int) -> bool:
        """Coloca una ficha si la casilla está vacía."""
        
        if board.board[row][col] != 0:
            return False
        board.board[row][col] = player_id
        board.player_positions[player_id].add((row,col))
        return True
        
    @staticmethod 
    def remove_piece(board: HexBoard, row: int, col: int, player_id: int) -> bool:
        """Remueve una ficha del tablero."""
        
        if board.board[row][col] == 0:
            return False
        board.board[row][col] = 0
        board.player_positions[player_id].remove((row,col))
        return True
    
    @staticmethod
    def check_connection(board: HexBoard, player_id: int) -> bool:
        """Verifica si el jugador ha conectado sus dos lados"""
        player_positions = []
        for i in range(board.size):
            for j in range(board.size):
                if (board.board[i][j] == player_id):
                    player_positions.append((i, j))
        
        return dfs(player_positions, player_id, board.size)
