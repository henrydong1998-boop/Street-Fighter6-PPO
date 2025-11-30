
import cv2
from ultralytics import YOLO
from wincam import DXCamera
from env.input_manager import InputManager, filter_input
from env.util import update_buffer_svd, random_projection
import torch
import numpy as np



# model = YOLO('./yolo/best.pt')
window_title = 'Street Fighter 6'
# print(gw.getAllTitles())


# embed_layers = [10]
# input_manager = InputManager("./mai_combos.json")
#TODO: implement: rl_model = RLModel()

def CV_test(model,model2,input_manager):
    with DXCamera(0, 0, 1920, 1080, fps=30) as camera:
        buffer = torch.zeros(0, 256)
        
        frame, _ = camera.get_bgr_frame()
        results = model.predict(frame, imgsz=256, conf=0.5)
        # embed = model(frame, embed=embed_layers)[0]
        # res = results[0]
        # embedding = res.embeddings[0]
        projectile_bbox = []
        opponent_bbox = []
        actor_bbox = []
        actor_state = -1
        opponent_state = -1
        projectile_state = -1

        for res in results:
            print(res.boxes)
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].int().tolist()
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                frame = np.ascontiguousarray(frame)
                frame = frame.astype(np.uint8, copy=False)
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                if cls < 10:
                    opponent_state = cls
                    actor_bbox.append([x1, x2, y1, y2])
                elif cls >= 10 and cls < 20:
                    actor_state = cls
                    opponent_bbox.append([x1, x2, y1, y2])
                else:
                    projectile_state = cls
                    projectile_bbox.append([x1, x2, y1, y2])
                    
                    
        embed_layers = [10]
        embed = model2.predict(frame, embed=embed_layers)[0]  
        
        # buffer, cache32 = update_buffer_svd(buffer=buffer, new_embed=embed, window_size=6, out_dim=32)
        
        embed = random_projection(embed=embed, out_dim=64, file="random_projection_256_64.npy")
        

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
    print(actor_state)
    CV_return=np.concatenate([np.array(actor_state).flatten(), np.array(opponent_state).flatten(), \
        np.array(projectile_state).flatten(), np.array(actor_bbox).flatten(),np.array(opponent_bbox).flatten(), \
        np.array(projectile_bbox).flatten()])
    return CV_return, actor_state, opponent_state