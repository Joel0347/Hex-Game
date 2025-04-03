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
            next_move, _ = self.minimax(board, True)
        return next_move
    
    def minimax(self, board: HexBoard, level_parity=True):
        opponent_id = 1 if self.player_id == 2 else 2
        
        if board.check_connection(opponent_id):
            return (-1, -1), 0
        elif board.check_connection(self.player_id):
            return (-1, -1), 1
        
        states = board.get_possible_moves()
        states_values = [0 for _ in states]
        
        if (level_parity):
            id = self.player_id
        else:
            id = opponent_id
        
        for i in range(len(states)):
            new_board = board.clone()
            new_board.place_piece(states[i][0], states[i][1], id)
            _, state_value = self.minimax(new_board, not level_parity)
            states_values[i] = state_value
            
        if (level_parity):
            state_value = max(states_values)
        else:
            state_value = min(states_values)
            
        return states[states_values.index(state_value)], state_value
        