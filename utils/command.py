class Command:
    def __init__(self) -> None:
        self.pick_move = None
        self.approach_move = None
        self.place_move = None
        self.return_move = None
        self.crate_picked_callback = None
        self.crate_placed_callback = None

    def set_pick_move(self, move):
        self.pick_move = move

    def set_approach_move(self, move):
        self.approach_move = move

    def set_place_move(self, move):
        self.place_move = move

    def set_return_move(self, move):
        self.return_move = move

    def set_crate_picked_callback(self, func):
        self.crate_picked_callback = func

    def set_crate_placed_callback(self, func):
        self.crate_placed_callback = func
