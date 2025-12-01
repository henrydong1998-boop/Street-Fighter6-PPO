from wincam import DXCamera
import cv2
import time
import numpy as np

# assumes bgr_frame of 1920x1080 resolution
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

    HSV_TOLERANCE = 10
    MIN_HSV = np.array([0, 0, 0])
    MAX_HSV = np.array([255, 255, 255])

    MIN_HSV_ACTOR = np.max([MIN_HSV, np.min([BEGIN_HSV_ACTOR, END_HSV_ACTOR], axis=0) - HSV_TOLERANCE], axis=0)
    MAX_HSV_ACTOR = np.min([MAX_HSV, np.max([BEGIN_HSV_ACTOR, END_HSV_ACTOR], axis=0) + HSV_TOLERANCE], axis=0)

    MIN_HSV_OPPONENT = np.max([MIN_HSV, np.min([BEGIN_HSV_OPPONENT, END_HSV_OPPONENT], axis=0) - HSV_TOLERANCE], axis=0)
    MAX_HSV_OPPONENT = np.min([MAX_HSV, np.max([BEGIN_HSV_OPPONENT, END_HSV_OPPONENT], axis=0) + HSV_TOLERANCE], axis=0)

    MIN_HSV_CRITICAL = np.max([MIN_HSV, np.min([BEGIN_HSV_CRITICAL, END_HSV_CRITICAL], axis=0) - HSV_TOLERANCE], axis=0)
    MAX_HSV_CRITICAL = np.min([MAX_HSV, np.max([BEGIN_HSV_CRITICAL, END_HSV_CRITICAL], axis=0) + HSV_TOLERANCE], axis=0)

    hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    actor_health_bar = hsv_frame[HEALTH_BAR_Y: HEALTH_BAR_Y + 1, MIN_X_ACTOR : MAX_X_ACTOR, : ]
    opponent_health_bar = hsv_frame[HEALTH_BAR_Y: HEALTH_BAR_Y + 1, MIN_X_OPPONENT : MAX_X_OPPONENT, : ]
    actor_healthbar_len = actor_health_bar.shape[1]
    opponent_healthbar_len = opponent_health_bar.shape[1]

    actor_health_normal = np.count_nonzero(cv2.inRange(actor_health_bar, MIN_HSV_ACTOR, MAX_HSV_ACTOR))
    opponent_health_normal = np.count_nonzero(cv2.inRange(opponent_health_bar, MIN_HSV_OPPONENT, MAX_HSV_OPPONENT))
    actor_health_critical = np.count_nonzero(cv2.inRange(actor_health_bar, MIN_HSV_CRITICAL, MAX_HSV_CRITICAL))
    opponent_health_critical = np.count_nonzero(cv2.inRange(opponent_health_bar, MIN_HSV_CRITICAL, MAX_HSV_CRITICAL))

    actor_health = max(actor_health_normal, actor_health_critical) / actor_healthbar_len
    opponent_health = max(opponent_health_normal, opponent_health_critical) / opponent_healthbar_len
    return (actor_health, opponent_health)

# camera = DXCamera(0, 32, 1920, 1080, fps=30)
class CameraHolder():
    def __init__(self):
        self.camera = DXCamera(0, 32, 1920, 1080, fps=30)

    def get_camera(self):
        return self.camera
