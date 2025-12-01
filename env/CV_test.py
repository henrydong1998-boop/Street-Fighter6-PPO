import cv2
from ultralytics import YOLO
from wincam import DXCamera
from env.util import update_buffer_svd, random_projection
import torch
import numpy as np
import cv2
import time

def extract_health_info(bgr_frame: np.ndarray) -> tuple[float, float]:
    start_time = time.time()
    HSV_TOLERANCE = 10
    HEALTH_BAR_Y = 91
    MIN_X_ACTOR, MAX_X_ACTOR = (236, 1127)
    MIN_HSV_ACTOR = (339 - HSV_TOLERANCE, 87 - HSV_TOLERANCE, 49 - HSV_TOLERANCE)
    MAX_HSV_ACTOR = (341 + HSV_TOLERANCE, 100, 86 + HSV_TOLERANCE)
    MIN_X_OPPONENT, MAX_X_OPPONENT = (1431, 2317)
    MIN_HSV_OPPONENT = (207 + HSV_TOLERANCE, 83 - HSV_TOLERANCE, 55 - HSV_TOLERANCE)
    MAX_HSV_OPPONENT = (224 + HSV_TOLERANCE, 93 + 7, 73 + HSV_TOLERANCE)

    hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    actor_health_bar = hsv_frame[HEALTH_BAR_Y, MIN_X_ACTOR : MAX_X_ACTOR, : ]
    opponent_health_bar = hsv_frame[HEALTH_BAR_Y, MIN_X_OPPONENT : MAX_X_OPPONENT, : ]
    actor_health_mask = cv2.inRange(actor_health_bar, MIN_HSV_ACTOR, MAX_HSV_ACTOR)
    opponent_health_mask = cv2.inRange(opponent_health_bar, MIN_HSV_OPPONENT, MAX_HSV_OPPONENT)
    end_time = time.time()
    print(f"function took {end_time - start_time} seconds")
    pass

# TODO: initializing camera each time function calls is stupid and ineffecient, initialize outside and pass
def CV_test(model, model2):
    with DXCamera(0, 32, 1920, 1080, fps=30) as camera:
        buffer = torch.zeros(0, 256)

        frame, _ = camera.get_bgr_frame()
        results = model.predict(frame, imgsz=256, conf=0.5)
        # embed = model(frame, embed=embed_layers)[0]
        # res = results[0]
        # embedding = res.embeddings[0]
        projectile_bbox = torch.zeros((1, 4))
        opponent_bbox = torch.zeros((1, 4))
        actor_bbox = torch.zeros((1, 4))
        actor_state = -1
        opponent_state = -1
        projectile_state = -1

        for res in results:
            # print(res.boxes)
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].int().tolist()
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                frame = np.ascontiguousarray(frame)
                frame = frame.astype(np.uint8, copy=False)
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
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

        # buffer, cache32 = update_buffer_svd(buffer=buffer, new_embed=embed, window_size=6, out_dim=32)

        embed = random_projection(embed=embed, out_dim=64, file="random_projection_256_64.npy")
        # print(embed)

        # TODO: implement pseudocode
        # 
        # 可以获得的变量有 actor_state, opponent_state, projectile_state, actor_bbox, opponent_bbox, projectile_bbox,embed
        # *_bbox 是2维数组，Nx4，表示检测到的该类别物体的边界框,绝大多数情况下N=1，可以拿到对手和自己的bbox 一共 8个值
        # *_state 是整数，表示检测到的该类别物体的状态
        # embed 是embedding向量，可以作为RL模型的输入之一 [64维]
        # 
        # predictions_unfiltered = rl_model.predict(results)
        # predictions = filter_input(actor_state, opponent_state) * prediction_unfiltered
        # prediction = argmax(predictions)
        # somehow update rl_model with filter_input mask
        # input_manager.update_facing(actor_bbox, opponent_bbox)
        # input_manager.accept_prediction(prediction)
        # input_manager.output_actions()

    # cv2.destroyAllWindows()
    # print(actor_state)
    CV_return=np.concatenate([np.array(actor_state).flatten(),
                              np.array(opponent_state).flatten(),
                              np.array(projectile_state).flatten(),
                              np.array(actor_bbox).flatten(),
                              np.array(opponent_bbox).flatten(),
                              np.array(projectile_bbox).flatten(),
                              embed.cpu().numpy().flatten()])

    # print(f"cv obs: {CV_return}")
    return CV_return, actor_state, opponent_state, actor_bbox, opponent_bbox