from numpy import random
from player_def import Player
from board import HexBoard

class AI_Player(Player):
    def play(self, board):
        if (len(board.get_possible_moves()) == board.size ** 2):
            row = random.randint(0, board.size - 1)
            col = random.randint(0, board.size - 1)
            next_move = (row, col)
        else:
            next_move, _ = self.minimax(board)
        return next_move
    
    def minimax(
        self, board: HexBoard, 
        level_parity=True, 
        alpha=-float('inf'), 
        betha=float('inf')
        ) -> tuple[tuple[int, int], int]:
        
        if board.check_connection(3 - self.player_id):
            return (-1, -1), 0
        elif board.check_connection(self.player_id):
            return (-1, -1), 1
        
        states = board.get_possible_moves()
        next_move = states[0]
        id = self.player_id if level_parity else 3 - self.player_id
        
        for st in states:
            AI_Player.place_piece(board, st[0], st[1], id)
            
            if level_parity:
                state_value = self.minimax(board, not level_parity, alpha=alpha)[1] 
            else:
                state_value = self.minimax(board, not level_parity, betha=betha)[1]

            AI_Player.remove_piece(board, st[0], st[1], id)
            
            if level_parity and alpha < state_value:
                alpha = state_value
                next_move = st
            elif (not level_parity) and betha > state_value:
                betha = state_value
                next_move = st
            
            if betha <= alpha:
                result = betha if level_parity else alpha
                return (-1, -1), result
            
        result = alpha if level_parity else betha
        
        return next_move, result
    
    
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