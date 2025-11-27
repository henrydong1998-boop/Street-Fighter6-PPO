import torch
from enum import Enum
import json


class InputFilter:
    pass

class InputKeys(Enum):
        LPunch = "U"
        MPunch = "I"
        HPunch = "O"
        Drive = "P"
        LKick = "H"
        MKick = "J"
        HKick = "K"
        DriveImpact = "L"
        Left = "A"
        Down = "S"
        Right = "D"
        Up = " "

class InputManager:
    input_list = []
    curr_combo = []
    combo_lists = []
    hold_keys = []
    frame_time = 1000 / 60

    def config_read(self, config_file: str):
        with open(config_file, "r") as file:
            self.combo_lists = json.load(file)
        return

    def __init__(self, config_file_path):
        self.config_read(config_file_path)
        pass

    class InputClass(Enum):
        combo1 = 1
        combo2 = 2
        combo3 = 3
        combo4 = 4
        combo5 = 5
        combo6 = 6
        combo7 = 7
        super1 = 8
        super2 = 9
        super3 = 10
        guard = 11
        drive = 12
        move_forward = 13
        move_backward = 14
        guard_lower = 15
        throw = 16

    def is_combo_action(self, action: InputClass | None):
        if action is None:
            return False
        if action in range(1, 8):
            return True
        return False

    def accept_prediction(self, prediction_class: int):
        prediction = self.InputClass(prediction_class)
        self.input_list.append(prediction)
        return

    def output_action(self):
        if len(self.input_list) == 0:
            return
        last_action = self.input_list[-1]
        if not self.is_combo_action(last_action):
            self.action_keys = self.input_list[last_action]
            self.