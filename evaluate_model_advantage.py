import os
from argparse import ArgumentParser
import cv2
import numpy as np


def extract_health_info(bgr_frame: np.ndarray) -> tuple[float, float]:
    HEALTH_BAR_Y = 67
    MIN_X_ACTOR, MAX_X_ACTOR = (177, 845)
    MIN_X_OPPONENT, MAX_X_OPPONENT = (1073, 1737)

    BEGIN_HSV_ACTOR = np.array([171, 255, 126])
    END_HSV_ACTOR = np.array([169, 222, 219])

    BEGIN_HSV_OPPONENT = np.array([104, 237, 186])
    END_HSV_OPPONENT = np.array([112, 215, 139])

    BEGIN_HSV_CRITICAL = np.array([30, 143, 253])
    END_HSV_CRITICAL = np.array([24, 169, 253])

    HSV_TOLERANCE = 20
    MIN_HSV = np.array([0, 0, 0])
    MAX_HSV = np.array([255, 255, 255])

    MIN_HSV_ACTOR = np.max([MIN_HSV, np.min([BEGIN_HSV_ACTOR, END_HSV_ACTOR], axis=0) - HSV_TOLERANCE], axis=0)
    MAX_HSV_ACTOR = np.min([MAX_HSV, np.max([BEGIN_HSV_ACTOR, END_HSV_ACTOR], axis=0) + HSV_TOLERANCE], axis=0)

    MIN_HSV_OPPONENT = np.max([MIN_HSV, np.min([BEGIN_HSV_OPPONENT, END_HSV_OPPONENT], axis=0) - HSV_TOLERANCE], axis=0)
    MAX_HSV_OPPONENT = np.min([MAX_HSV, np.max([BEGIN_HSV_OPPONENT, END_HSV_OPPONENT], axis=0) + HSV_TOLERANCE], axis=0)

    MIN_HSV_CRITICAL = np.max([MIN_HSV, np.min([BEGIN_HSV_CRITICAL, END_HSV_CRITICAL], axis=0) - HSV_TOLERANCE], axis=0)
    MAX_HSV_CRITICAL = np.min([MAX_HSV, np.max([BEGIN_HSV_CRITICAL, END_HSV_CRITICAL], axis=0) + HSV_TOLERANCE], axis=0)

    hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    actor_health_bar = np.flip(hsv_frame[HEALTH_BAR_Y: HEALTH_BAR_Y + 1, MIN_X_ACTOR : MAX_X_ACTOR, : ], axis=1)
    opponent_health_bar = hsv_frame[HEALTH_BAR_Y: HEALTH_BAR_Y + 1, MIN_X_OPPONENT : MAX_X_OPPONENT, : ]
    actor_healthbar_len = actor_health_bar.shape[1]
    opponent_healthbar_len = opponent_health_bar.shape[1]

    actor_health_normal_idx = np.flatnonzero(cv2.inRange(actor_health_bar, MIN_HSV_ACTOR, MAX_HSV_ACTOR))
    actor_health_normal = actor_health_normal_idx[-1] if actor_health_normal_idx.shape[0] > 0 else 0
    opponent_health_normal_idx = np.flatnonzero(cv2.inRange(opponent_health_bar, MIN_HSV_OPPONENT, MAX_HSV_OPPONENT))
    opponent_health_normal = opponent_health_normal_idx[-1] if opponent_health_normal_idx.shape[0] > 0 else 0
    actor_health_critical_idx = np.flatnonzero(cv2.inRange(actor_health_bar, MIN_HSV_CRITICAL, MAX_HSV_CRITICAL))
    actor_health_critical = actor_health_critical_idx[-1] if actor_health_critical_idx.shape[0] > 0 else 0
    opponent_health_critical_idx = np.flatnonzero(cv2.inRange(opponent_health_bar, MIN_HSV_CRITICAL, MAX_HSV_CRITICAL))
    opponent_health_critical = opponent_health_critical_idx[-1] if opponent_health_critical_idx.shape[0] > 0 else 0

    actor_health =  max(actor_health_normal, actor_health_critical) / actor_healthbar_len
    opponent_health = max(opponent_health_normal, opponent_health_critical) / opponent_healthbar_len
    return (actor_health, opponent_health)

def extract_match_info(start_times: list[int] | None, end_times: list[int], advantages: list[float]) -> str:
    win_rate = len(list(filter(lambda advantages: advantages > 0, advantages))) / len(advantages)
    average_advantages = sum(advantages) / len(advantages)
    match_durations = []
    average_duration = "N/a"
    if start_times is not None:
        assert len(start_times) == len(end_times)
        for i in range(len(start_times)):
            match_durations.append(end_times[i] - start_times[i])
        average_duration = sum(match_durations) / len(match_durations)
        return f"win rate: [{win_rate: 2.0%}], " + \
               f"average advantage: [{average_advantages: 2.0%}], " + \
               f"average duration: [{average_duration: 2.1f}s]"
    return f"win rate: [{win_rate: 2.0%}], "+ \
           f"average advantage: [{average_advantages: 2.0%}], "+ \
           f"average duration: [{average_duration}]"

if __name__ == "__main__":
    args_parser = ArgumentParser()
    args_parser.add_argument("-s","--source", dest="source", default=None)
    args_parser.add_argument("-d","--dest", dest="dest", default=None)
    args_parser.add_argument("--start_times", dest="start_times", default=None)
    args = args_parser.parse_args()
    source_path = args.source
    source_files = [source_path]
    start_times = None
    end_times = []
    health_advantages = []

    if os.path.isdir(source_path):
        source_files = os.listdir(source_path)
        source_files = filter(lambda filename: filename.endswith((".png", ".jpg", ".jpeg")), source_files)
        source_files = map(lambda filename: os.path.join(source_path, filename), source_files)

    for source_file in source_files:
        assert(source_file and source_file.endswith((".png", ".jpg", ".jpeg")))
        source_bgr_frame = cv2.imread(source_file)
        cropped_bgr_frame = source_bgr_frame[32: 1080 + 32, 0: 1920, :]
        actor_health, opponent_health = extract_health_info(cropped_bgr_frame)
        health_advantage = actor_health - opponent_health
        health_advantages.append(health_advantage)
        print(f"{source_file} >>> Actor Health: [{actor_health: 2.0%}], " + \
              f"Opponent Health: [{opponent_health: 2.0%}], " + \
              f"Health Advantage:[{health_advantage: 2.0%}]")
        filename = str(os.path.basename(source_file))
        name, extension = filename.split(".")
        minutes, seconds = name.split("_")
        end_times.append(int(minutes) * 60 + int(seconds))

    if args.start_times is not None:
        lines = list(open(args.start_times))
        start_times = []
        for line in lines:
            minutes, seconds = line.split("_")
            start_times.append(int(minutes) * 60 + int(seconds))
    match_info = f"{source_path} >>> " + extract_match_info(start_times, end_times, health_advantages)
    print(match_info)

    if args.dest is not None:
        with open(args.dest, "a") as file:
            file.write(match_info + "\n")