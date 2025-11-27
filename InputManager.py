from enum import Enum
import json
import time
import keyboard

def filter_input(actions_tag_player, actions_tag_opponent) -> list[bool]:
    pass

InputKeys = {
    "LPunch":      "U",
    "MPunch":      "I",
    "HPunch":      "O",
    "DriveGuard":  "P",
    "Kick":        "H",
    "MKick":       "J",
    "HKick":       "K",
    "DriveImpact": "L",
    "Left":        "A",
    "Down":        "S",
    "Right":       "D",
    "Up":          " "
}

class InputManager:
    NUM_CLASSES = 16
    input_list: list["InputClass"] = []
    combo_lists = {}
    hold_keys = []
    curr_combo_action = None
    curr_combo_index = -1
    frame_time = None
    facing_right = True

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

    def __init__(self, config_file_path: str, frame_time_seconds: float = 1 / 60):
        assert len(self.InputClass) == self.NUM_CLASSES
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
        return self.combo_list[action.name]

    def is_combo_action(self, action: InputClass | None) -> bool:
        if action is None:
            return False
        if action in range(1, 8):
            return True
        return False

    def accept_prediction(self, prediction: int) -> None:
        assert prediction >= 0 and prediction <= self.NUM_CLASSES
        prediction_class = self.InputClass(prediction)
        self.input_list.append(prediction_class)
        return

    # return True for facing right, False for facing left
    def update_facing(self, actor_bbox: list[float], opponent_bbox: list[float]) -> bool:
        x1, x2, y1, y2 = actor_bbox
        x3, x4, y3, y4 = opponent_bbox
        self.facing_right = (x1 + x2) < (x3 + x4)
        return self.facing_right

    def output_actions(self) -> None:
        if len(self.input_list) == 0:
            return
        last_action = self.input_list[-1]
        curr_action_dict = self.get_action_dict(last_action)
        if (not self.is_combo_action(last_action)) or \
           (not self.is_combo_action(self.curr_combo_action)) or \
           (last_action != self.curr_combo_action) or \
           (curr_action_dict["combo_breaks"][0] == -1):
            self.curr_combo_name = last_action
            self.curr_combo_index = 0
            actions_until = curr_action_dict["combo_breaks"][self.curr_combo_index]
            actions = curr_action_dict["actions"][0: actions_until]
            wait_times = curr_action_dict["wait_frames"][0: actions_until]
            self.act_and_wait(actions, wait_times)
            self.release_all_keys()
            return

        if self.curr_combo_index == len(curr_action_dict["combo_breaks"]):
            #TODO: Finish function
            pass

    def release_all_keys(self) -> None:
        for input_key in InputKeys.values():
            if keyboard.is_pressed(input_key):
                keyboard.release(input_key)
        return

    def act_and_wait(self, actions: list[str], wait_frames: list[int]) -> None:
        assert len(actions) == len(wait_frames)
        for index in range(len(actions)):
            action = actions[index]
            wait_frame = wait_frames[index]
            action_keys = action.split("_")

            for action_key in action_keys:
                hold = (action[-1] == "H")
                release = (action[-1] == "R")
                input_key = ""
                if hold or release:
                    action_key = action_key[: -2]

                if action_key == "Front" or action_key == "Back":
                    input_key = InputKeys["Left"] if (action_key == "Front") != (self.facing_right) else InputKeys["Right"]
                else:
                    input_key = InputKeys[action]

                if hold and not keyboard.is_pressed(input_key):
                    keyboard.press(input_key)
                if release and keyboard.is_pressed(input_key):
                    keyboard.release(input_key)
                if (not hold) and (not release):
                    keyboard.press_and_release(input_key)

            if wait_frame != 0:
                time.sleep(wait_frame * self.frame_time)
        return

if __name__ == "__main__":
    manager = InputManager("./mai_combos.json")
    for key, value in manager.combo_lists.items():
        print(f"key: {key}")
        print(f"value: {value}")