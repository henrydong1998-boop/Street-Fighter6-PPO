import cv2
from ultralytics import YOLO
from wincam import DXCamera
from env.util import update_buffer_svd, random_projection
import torch
import numpy as np
import cv2
import time

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

def CV_test(camera, model, model2):
    buffer = torch.zeros(0, 256).cuda()

    frame, _ = camera.get_bgr_frame()
    results = model.predict(frame, imgsz=256, conf=0.5)
    projectile_bbox = torch.zeros((1, 4))
    opponent_bbox = torch.zeros((1, 4))
    actor_bbox = torch.zeros((1, 4))
    actor_state = -1
    opponent_state = -1
    projectile_state = -1
    actor_health, opponent_health = extract_health_info(bgr_frame=frame)

    for res in results:
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].int().tolist()
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            frame = np.ascontiguousarray(frame)
            frame = frame.astype(np.uint8, copy=False)
            if cls < 10:
                opponent_state = cls
                actor_bbox = torch.tensor([x1, x2, y1, y2])
            elif cls >= 10 and cls < 20:
                actor_state = cls
                opponent_bbox = torch.tensor([x1, x2, y1, y2])
            else:
                projectile_state = cls
                projectile_bbox = torch.tensor([x1, x2, y1, y2])


    embed_layers = [10]
    embed = model2.predict(frame, embed=embed_layers)[0]  

    buffer, cache32 = update_buffer_svd(buffer=buffer, new_embed=embed, window_size=4, out_dim=32)

    embed = random_projection(embed=embed, out_dim=128, file="random_projection_256_64.npy")

    CV_return=np.concatenate([np.array([actor_health, opponent_health]).flatten(),
                              np.array(actor_state).flatten(),
                              np.array(opponent_state).flatten(),
                              np.array(projectile_state).flatten(),
                              np.array(actor_bbox).flatten(),
                              np.array(opponent_bbox).flatten(),
                              np.array(projectile_bbox).flatten(),
                              cache32.cpu().numpy().flatten(),
                              embed.cpu().numpy().flatten()])

    return CV_return, actor_state, opponent_state, actor_bbox, opponent_bbox, actor_health, opponent_health