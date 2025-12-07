from extract_health_bar import extract_health_info, CameraHolder
import time
import cv2

if __name__ == "__main__":
    time.sleep(5)
    counter = 0
    camera_holder = CameraHolder()
    while True:
        camera = camera_holder.get_camera()
        frame, _ = camera.get_bgr_frame()
        time_begin = time.time()
        actor_health, opponent_health = extract_health_info(frame)
        total_time = time.time() - time_begin
        print(f"actor health is {actor_health}, and opponent health is {opponent_health}, took{total_time * 1000} ms")
        # time.sleep(5)
        # cv2.imwrite(f"captureimg{counter}.png", frame)
        counter += 1
