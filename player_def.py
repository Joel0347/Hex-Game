from board import HexBoard
from abc import ABC, abstractmethod

class Player(ABC):
    def __init__(self, player_id: int):
        self.player_id = player_id  # Tu identificador (1 o 2)

    @abstractmethod
    def play(self, board: HexBoard) -> tuple:
        pass  
