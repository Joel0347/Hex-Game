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
        
        if not board.board[board.size // 2][board.size // 2]:
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
        
        if game_phase > 0.70:
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
        
        if AI_Player.check_connection(board, 3 - self.player_id)[0]:
            return (), -1000
        elif AI_Player.check_connection(board, self.player_id)[0]:
            return (), 1000
        
        if (not depth):
            return (), self.heuristic(board, id)
        
        possible_moves = board.get_possible_moves()
        possible_moves.sort(key=lambda move: (abs(move[0]-board.size//2) + abs(move[1]-board.size//2)))
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

    def simulate(self, board: HexBoard, player_id: int, player_on_turn: int, depth=5) -> int:
        """Simula cada jugada de manera aleatoria hasta que gane alguien"""
        
        if AI_Player.check_connection(board, player_id)[0]:
            result = float('inf') if depth != float('inf') else 1
        elif AI_Player.check_connection(board, 3 - player_id)[0]:
            result = -float('inf') if depth != float('inf') else 0
        elif not depth:
            return self.heuristic(board, player_id)
        else:
            possible_moves = board.get_possible_moves()
            row, col = possible_moves[random.choice(len(possible_moves))]
            AI_Player.place_piece(board, row, col, player_on_turn)
            result = self.simulate(board, player_id, 3 - player_on_turn, depth - 1)
            AI_Player.remove_piece(board, row, col, player_on_turn)

        return result

    def monte_carlo_method(self, board: HexBoard, player_id: int, simulations=1000) -> tuple[int, int]:
        """
        Método de monte carlo para simular las jugadas hasta que alguien gane y 
        contar la cantidad de victorias en cada jugada realizada por simulación, 
        así se escoge la que más victorias haya producido
        """
        
        possible_moves = board.get_possible_moves()
        move_score_counts = {move: 0 for move in possible_moves}
        move_played_count = {move: 0 for move in possible_moves}
        depth = 5 if board.size > 7 else float('inf')
        simulations = 2000 if board.size < 9 else 1000

        while simulations > 0:
            move = possible_moves[random.randint(0, len(possible_moves))]
            move_played_count[move] += 1
            AI_Player.place_piece(board, move[0], move[1], player_id)
            move_score_counts[move] += self.simulate(board, player_id, 3 - player_id, depth)
            AI_Player.remove_piece(board, move[0], move[1], player_id)
            
            simulations -= 1
        
        return max(
            possible_moves, key=lambda move: move_score_counts[move] / move_played_count[move] 
                if move_played_count[move] > 0 else 0
        )

    def heuristic(self, board: HexBoard, player_id: int) -> int:
        """
        Heurística que evalúa la menor cantidad de movimientos para ganar.
        Prioriza jugadas ganadoras o bloquear al oponente.
        """
        
        player_path = self.shortest_path(board, player_id)
        opponent_id = 3 - player_id
        opponent_path = self.shortest_path(board, opponent_id)
        
        if player_path == 0:
            return float('inf')
        if opponent_path == 0:
            return -float('inf')
        return opponent_path - player_path

    @staticmethod
    def shortest_path(board: HexBoard, player_id: int) -> int:
        size = board.size
        distance = [[float('inf')] * size for _ in range(size)]
        dq = deque()
        
        if player_id == 1:
            start_nodes = [(i, 0) for i in range(size)]
            end_condition = lambda x, y: y == size - 1
        else:
            start_nodes = [(0, i) for i in range(size)]
            end_condition = lambda x, y: x == size - 1
        
        for x, y in start_nodes:
            if board.board[x][y] == player_id:
                distance[x][y] = 0
                dq.appendleft((x, y))
            elif board.board[x][y] == 0:
                distance[x][y] = 1
                dq.append((x, y))
        
        directions = [ (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0) ]
        min_dist = float('inf')   
        
        while dq:
            x, y = dq.popleft()
            
            if end_condition(x, y):
                min_dist = min(min_dist, distance[x][y])
                continue
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    if board.board[nx][ny] == 3 - player_id:
                        continue
                    new_dist = distance[x][y] + (0 if board.board[nx][ny] == player_id else 1)
                    if new_dist < distance[nx][ny]:
                        distance[nx][ny] = new_dist
                        if new_dist == distance[x][y]:
                            dq.appendleft((nx, ny))
                        else:
                            dq.append((nx, ny))
        
        return min_dist

    
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
            if AI_Player.check_connection(board, player_id)[0]:
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
            if (AI_Player.check_connection(board, player_id)[0]):
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
        
        size = board.size
        win_area = [
            (i, j) for i in range(size) for j in [0, size - 1]
            ] if player_id == 1 else [
                (i, j) for j in range(size) for i in [0, size - 1]
                ]
            
        player_pos = [
            (i, j) for i, row in enumerate(board.board) 
            for j, value in enumerate(row) if value == player_id
            ]
        
        opponent_pos = [
            (i, j) for i, row in enumerate(board.board) 
            for j, value in enumerate(row) if value == 3 - player_id
            ]
        
        player_pos_reduced = [
            (pos[0] - 1, pos[1] - 1) for pos in player_pos 
                if pos[0] > 0 and pos[1] > 0 and pos[0] < size - 1 and pos[1] < size - 1
            ]
        opponent_pos_reduced = [
            (pos[0] - 1, pos[1] - 1) for pos in opponent_pos 
                if pos[0] > 0 and pos[1] > 0 and pos[0] < size - 1 and pos[1] < size - 1
            ]
        positions = {player_id: player_pos_reduced, (3-player_id): opponent_pos_reduced}
        inside_board = AI_Player.build_reduced_board(board.size - 2, positions)
        won, (start, end) = AI_Player.check_connection(inside_board, player_id)
        
        if not won:
            return False
        
        adj_start = [
            (row, col) for (row, col) in win_area 
            if not board.board[row][col] and AI_Player.is_adjacent(board, start, (row, col))
            ]
        adj_end = [
            (row, col) for (row, col) in win_area 
            if not board.board[row][col] and AI_Player.is_adjacent(board, end, (row, col))
            ]
        
        return len(adj_end) >= 2 and len(adj_start) >= 2
        
    @staticmethod
    def is_adjacent(board: HexBoard, place: tuple[int, int], to_check: tuple[int, int]) -> bool:
        adj = [(0,1), (0,-1), (1,-1), (1,0), (-1,1), (-1,0)]
        
        for dx, dy in adj:
            if place[0] + dx == to_check[0] and place[1] + dy == to_check[1]:
                return True
        return False 
        
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
    def check_connection(
        board: HexBoard, player_id: int
        ) -> tuple[bool, tuple[tuple[int, int], tuple[int, int]]]:
        """
        Verifica si el jugador ha conectado sus dos lados y 
        devuelve los extremos ganadores
        """
        player_positions = []
        for i in range(board.size):
            for j in range(board.size):
                if (board.board[i][j] == player_id):
                    player_positions.append((i, j))
        
        return dfs(player_positions, player_id, board.size)
