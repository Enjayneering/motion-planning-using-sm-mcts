import numpy as np


class RaceGame:
    """
    A simple piece of a racetrack defined as a grid of cells:
        rows: 1
        columns: 4
        winNumber: 2
    """

    def __init__(self):
        self.rows = 40
        self.columns = 11
        self.win = 2

    def get_init_board(self):
        b = np.zeros((self.rows,self.columns), dtype=np.int)
        return b

    def get_board_size(self):
        return (self.rows, self.columns)

    def get_action_size(self):
        return 5

    def has_legal_moves(self, board): # TODO: Change, so that collision is checked
        for index in range(self.columns):
            if board[index] == 0:
                return True
        return False

    def get_valid_moves(self, state, env):
        # All moves are invalid by default
        valid_moves = [0] * self.get_action_size()

        for index in range(self.columns):
            if board[index] == 0:
                valid_moves[index] = 1

        return valid_moves

    def is_win(self, board, player):
        count = 0
        for index in range(self.columns):
            if board[index] == player:
                count = count + 1
            else:
                count = 0

            if count == self.win:
                return True

        return False

    def get_reward_for_player(self, board, player):
        # return None if not ended, 1 if player 1 wins, -1 if player 1 lost

        if self.is_win(board, player):
            return 1
        if self.is_win(board, -player):
            return -1
        if self.has_legal_moves(board):
            return None

        return 0

    def get_canonical_board(self, board, player):
        return player * board


class Environment:
    def __init__(self):
        # static env information
        self.state_space = None
        self.state_space_obstacles = None
        self.center_line = None

        # dynamic env information
        self.joint_state = None
        self.joint_trajectory = None
        self.joint_control_policy = None


    def define_state_space(self, map, ):
        pass



maps = {
    'street_with_passage': """
    .......#.......
    ...............
    .......#.......
    .......#.......
    .......#.......
    .......#.......
    .......#.......
    """
}
