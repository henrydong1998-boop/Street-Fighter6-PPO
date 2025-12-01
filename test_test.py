from test_health_bar import extract_health_info, CameraHolder
import time
import cv2

if __name__ == "__main__":
    time.sleep(5)
    time_begin = time.time()
    counter = 0
    camera_holder = CameraHolder()
    while True:
        camera = camera_holder.get_camera()
        frame, _ = camera.get_bgr_frame()
        actor_health, opponent_health, total_time= extract_health_info(frame)
        print(f"actor health is {actor_health}, and oppnent health is {opponent_health}, function took {total_time}_ms")
        # time.sleep(5)
        # cv2.imwrite(f"captureimg{counter}.png", frame)
        counter += 1
        time_end = time.time()
        print(f"calling took {time_end - time_begin} seconds")
        time_begin = time_end