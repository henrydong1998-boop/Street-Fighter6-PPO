
import cv2
from ultralytics import YOLO
# cha老师说是Windows API
from py_screen_grab import ScreenGrabber


model = YOLO('./best.pt')
# results = model.predict('2025-11-14 23-20-46.mp4', save=True, imgsz=256, conf=0.5)
# for i, (name, module) in enumerate(model.model.named_modules()):
#     print(i, name, module)

window_title = "My Game Title"
grabber = ScreenGrabber().set_window(window_title).set_fps(30)


embed_layers = [10]

grabber.start_streaming()

for frame in grabber.stream():
    results = model.predict(source=frame, stream=False, embed=embed_layers, imgsz=416)
    res = results[0]
    embedding = res.embeddings[0]
    print("Embedding shape:", embedding.shape)

    for box, cls, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{int(cls)}:{conf:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("Haha", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

grabber.stop_streaming()
cv2.destroyAllWindows()