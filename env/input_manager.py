from enum import Enum
import json
import time
import keyboard
import torch
import numpy as np


InputKeys = {
    "LPunch":      "U",
    "MPunch":      "I",
    "HPunch":      "O",
    "DriveGuard":  "P",
    "LKick":       "H",
    "MKick":       "J",
    "HKick":       "K",
    "DriveImpact": "L",
    "Left":        "A",
    "Down":        "S",
    "Right":       "D",
    "Up":          " "
}

CVClassNames = {
    0: "opponent_attack",
    1: "opponent_attack_lower",
    2: "opponent_drive",
    3: "opponent_guard",
    4: "opponent_guard_lower",
    5: "opponent_hit",
    6: "opponent_jump",
    7: "opponent_neutral",
    8: "opponent_super",
    9: "opponent_throw",
    10: "player_attack",
    11: "player_attack_lower",
    12: "player_drive",
    13: "player_guard",
    14: "player_guard_lower",
    15: "player_hit",
    16: "player_jump",
    17: "player_neutral",
    18: "player_super",
    19: "player_throw",
    20: "projectile_opponent",
    21: "projectile_player"
}

CVClassToTags = {value: key for key, value in CVClassNames.items()}

class InputClass(Enum):
    combo1 =        1
    combo2 =        2
    combo3 =        3
    combo4 =        4
    combo5 =        5
    combo6 =        6
    combo7 =        7
    super1 =        8
    super2 =        9
    super3 =        10
    guard =         11
    guard_lower =   12
    drive_impact =  13
    move_forward =  14
    move_backward = 15
    throw =         16

def filter_input(actions_tag_player, actions_tag_opponent) -> list[bool]:
    #TODO: finish function
    action_filter = np.array([True for i in range(16)])
    if actions_tag_player < 0 or actions_tag_opponent <0 :
        return action_filter
    player_tag = CVClassNames[actions_tag_player]
    opponent_tag = CVClassNames[actions_tag_opponent]

    # inputclass begins at 1, whereas input indices begin at 0
    # print(InputClass.combo1.value - 1, InputClass.super3.value)
    if player_tag in ["player_guard", "player_guard_down"]:
        action_filter[InputClass.combo1.value - 1: InputClass.super3.value] = False
        # numpy good, list bad
    elif player_tag == "player_hit":
        action_filter[InputClass.combo1.value - 1: InputClass.super3.value] = False
        action_filter[InputClass.combo4.value - 1] = True
        action_filter[InputClass.drive_impact.value - 1] = True
        action_filter[InputClass.move_forward.value - 1] = True

    if opponent_tag == "opponent_throw":
        action_filter = [False for i in range(16)]
        action_filter[InputClass.throw.value - 1] = True
        action_filter[InputClass.drive_impact.value - 1] = True
        action_filter[InputClass.guard.value - 1] = True
        action_filter[InputClass.guard_lower.value - 1] = True
    elif opponent_tag == "opponent_guard":
        action_filter[InputClass.guard.value - 1] = False
        action_filter[InputClass.move_forward.value - 1] = False
    elif opponent_tag == "opponent_attack_lower":
        action_filter[InputClass.guard.value - 1] = False

    return action_filter

class InputManager:
    NUM_CLASSES = 16
    input_list: list["InputClass"] = []
    combo_lists = {}
    curr_combo_action = None
    curr_combo_index = -1
    frame_time = None
    facing_right = True


    def __init__(self, config_file_path: str, frame_time_seconds: float = 1 / 60):
        assert len(InputClass) == self.NUM_CLASSES
        self.config_read(config_file_path)
        self.frame_time = frame_time_seconds

    def config_read(self, config_file: str) -> None:
        with open(config_file, "r") as file:
            combo_lists = json.load(file)
            for index in range(1, len(combo_lists) + 1):
                combo = combo_lists[f"{index}"]
                self.combo_lists[combo["name"]] = {"actions": combo["actions"],
                                                   "wait_frames": combo["wait_frames"],
                                                   "combo_breaks": combo["combo_breaks"]}
        return

    def get_action_dict(self, action: InputClass) -> dict:
        assert(action.name in self.combo_lists.keys())
        return self.combo_lists[action.name]

    def is_combo_action(self, action: InputClass | None) -> bool:
        if action is None:
            return False
        elif action.value in range(1, 8):
            return True
        return False

    def accept_prediction(self, prediction: int) -> None:
        assert prediction >= 0 and prediction <= self.NUM_CLASSES
        # print(prediction)
        prediction_class = InputClass(prediction + 1)
        self.input_list.append(prediction_class)
        # print(prediction_class)
        return

    # return True for facing right, False for facing left
    def update_facing(self, actor_bbox: torch.Tensor, opponent_bbox: torch.Tensor) -> bool:
        empty_bbox = torch.zeros((1, 4))
        if (torch.equal(actor_bbox, empty_bbox)) or (torch.equal(opponent_bbox, empty_bbox)):
            self.facing_right = True
            return self.facing_right

        x1, x2, y1, y2 = actor_bbox
        x3, x4, y3, y4 = opponent_bbox
        self.facing_right = (x1 + x2) > (x3 + x4) # wierd how it ended up being this
        return self.facing_right

    def output_actions(self) -> None:
        if len(self.input_list) == 0:
            return
        last_action = self.input_list[-1]
        print(f"current action: {last_action.name}")
        curr_action_dict = self.get_action_dict(last_action)
        # print(f"curr dict is {curr_action_dict}")
        if (not self.is_combo_action(last_action)) or \
           (not self.is_combo_action(self.curr_combo_action)) or \
           (last_action != self.curr_combo_action) or \
           (curr_action_dict["combo_breaks"][0] == -1):
            self.release_all_keys()
            # print("first action of combo")
            self.curr_combo_action = last_action
            self.curr_combo_index = 0
            actions_until = curr_action_dict["combo_breaks"][self.curr_combo_index]
            if actions_until == -1:
                actions_until = len(curr_action_dict["actions"])
            actions = curr_action_dict["actions"][0: actions_until]
            wait_times = curr_action_dict["wait_frames"][0: actions_until]
            self.act_and_wait(actions, wait_times)
            self.input_list = []
            return

        if self.curr_combo_index >= len(curr_action_dict["combo_breaks"]) - 1:
            # print(f"curr_combo_index was {self.curr_combo_index}, resetting to 0")
            self.curr_combo_index = 0
        else:
            self.curr_combo_index = self.curr_combo_index + 1
            # print(f"incrementing curr_combo_index to {self.curr_combo_index}")
        actions_from  = 0
        if self.curr_combo_index >= 1:
            actions_from = curr_action_dict["combo_breaks"][self.curr_combo_index - 1]
        actions_until = curr_action_dict["combo_breaks"][self.curr_combo_index]
        if actions_until == -1:
            actions_until = len(curr_action_dict["actions"])
        # print(f"performing actions from {actions_from} to {actions_until}")
        actions = curr_action_dict["actions"][actions_from: actions_until]
        wait_times = curr_action_dict["wait_frames"][actions_from: actions_until]
        self.act_and_wait(actions, wait_times)
        self.input_list = []
        return

    def release_all_keys(self) -> None:
        for input_key in InputKeys.values():
            if keyboard.is_pressed(input_key):
                keyboard.release(input_key)
        return

    def act_and_wait(self, actions: list[str], wait_frames: list[int]) -> None:
        # print(f"actions: {actions}")
        # print(f"wait_times: {wait_frames}")
        assert len(actions) == len(wait_frames)
        for index in range(len(actions)):
            action = actions[index]
            wait_frame = wait_frames[index]
            action_keys = action.split("_")
            to_release = []

            for action_key in action_keys:
                hold = (action_key[-1] == "H")
                release = (action_key[-1] == "R")
                input_key = ""
                if hold or release:
                    action_key = action_key[: -1]

                if action_key == "Front" or action_key == "Back":
                    is_towards_left = (action_key == "Front") != (self.facing_right)
                    input_key = InputKeys["Left"] if is_towards_left else InputKeys["Right"]
                else:
                    input_key = InputKeys[action_key]

                if hold and not keyboard.is_pressed(input_key):
                    keyboard.press(input_key)
                if release:
                    keyboard.release(input_key)
                if (not hold) and (not release):
                    keyboard.press(input_key)
                    to_release.append(input_key)

            if len(to_release)!= 0:
                time.sleep(self.frame_time)
                for key in to_release:
                    keyboard.release(key)

            if wait_frame != 0:
                time.sleep(wait_frame * self.frame_time)
        return
