
import cv2
from ultralytics import YOLO
from wincam import DXCamera
from input_manager import InputManager

model = YOLO('./best.pt')
# results = model.predict('2025-11-14 23-20-46.mp4', save=True, imgsz=256, conf=0.5)
# for i, (name, module) in enumerate(model.model.named_modules()):
#     print(i, name, module)

window_title = 'Street Fighter 6'
# print(gw.getAllTitles())


embed_layers = [10]
input_manager = InputManager("./mai_combos.py")

with DXCamera(0, 0, 1920, 1080, fps=30) as camera:
    while True:
        frame, _ = camera.get_bgr_frame()
        results = model.predict(frame, imgsz=256, conf=0.5)
        # embed = model(frame, embed=embed_layers)[0]
        # res = results[0]
        # embedding = res.embeddings[0]
        for res in results:
            print(res.boxes)
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].int().tolist()
                cls = int(b.cls[0])
                conf = float(b.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # TODO: implement pseudocode
        # rl_model = RLModel()
        # prediction = rl_model.predict(results)
        # input_manager.update_facing(actor_bbox, opponenet_bbox)
        # input_manager.accept_prediction(prediction)
        # input_manager.output_actions()


cv2.destroyAllWindows()