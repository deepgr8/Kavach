from ultralytics import YOLO
import cv2
import imutils
import pywhatkit
from ultralytics.utils.plotting import Annotator
from datetime import datetime
import threading
import time
import numpy as np

print("Working")
def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        t -= 1
    return True
model = YOLO('./runs/detect/train6/weights/best.pt')
# multiFrame = [
#     'rtsp://admin:admin123@10.0.199.106:554/cam/realmonitor?channel=3&subtype=0',
#     'rtsp://admin:admin123@10.0.199.106:554/cam/realmonitor?channel=4&subtype=0',
#     'rtsp://admin:admin123@10.0.199.106:554/cam/realmonitor?channel=5&subtype=0',
#     'rtsp://admin:admin123@10.0.199.106:554/cam/realmonitor?channel=6&subtype=0'
# ]
multiFrame=['http://192.168.83.168:4747/video','http://192.168.83.216:4747/video','http://192.168.83.124:4747/video']
cap = [cv2.VideoCapture(url) for url in multiFrame]
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('rtsp://admin:MGINLP@10.0.158.204:554/cam/realmonitor?channel=1&subtype=0')
while True:
    frames = []
    try:    
        for ca in cap:
            ret, frame = ca.read()
            frame = imutils.resize(frame, width=500, height=500)  # Resize frame to desired width and height
            frames.append(frame)
    except:
        continue
    if not frames:
        continue

    no_fight_detected = True
    for frame in frames:
        results = model.predict(source=frame)
        if not results or len(results) == 0:
            continue
        annotator = Annotator(frame)
        for result in results:
            detection_count = result.boxes.shape[0]
            bd_boxes = result.boxes
            for box in bd_boxes:
                b = box.xyxy[0]
                c = box.cls
                annotator.box_label(b, model.names[int(c)], color=(255, 0, 0), txt_color=(0, 0, 255))

            for i in range(detection_count):
                cls = int(result.boxes.cls[i].item())
                name = result.names[cls]
                print(name)
                # if name == "Violence":
                #     print("its violence detected")
                #     start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                #     cv2.putText(frame, "Timing " + str(start_time), (20, 650), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                #                 (255, 0, 0), 1, cv2.LINE_AA, False)
                # if name=="NonViolence":
                #     print("NonViolence detected")
                #     timer = threading.Thread(target=countdown, args=(4)) 
                #     if timer:
                #         print("countdown started")
                #         end_time =datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                #         f = open("ReportFile.txt","w+")
                #         f.write(f"Start timing of fight is{start_time}")
                #         f.write(f"Start timing of fight is{end_time}")
                #         f.close()
                #         pywhatkit.sendwhatmsg("+919755833563",
                #         "Violence Detected",
                #         18, 30)
        
    frame = annotator.result()
    frame = imutils.resize(frame, width=500)
    # frame = cv2.flip(frame, 1)
    # Split frames into two halves
    
    combined_frame = np.hstack(frames)
    # half_index = len(frames) // 2
    # first_half = frames[:half_index]
    # second_half = frames[half_index:]
    # combined_first_row = np.hstack(first_half)  # Combine frames in the first row
    # combined_second_row = np.hstack(second_half)  # Combine frames in the second row
    # combined_frame = np.vstack([combined_first_row, combined_second_row])  # Combine both rows vertically
    cv2.imshow("Combined Frames", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for ca in cap:
    ca.release()
cv2.destroyAllWindows()
