# from ultralytics import YOLO
# model = YOLO('./runs/detect/train5/weights/best.pt')
# model.predict(source=0,show=True)


from ultralytics import YOLO
import cv2
import time
import pywhatkit as w
import pyautogui
import keyboard as k
import imutils
import threading
from ultralytics.utils.plotting import Annotator
from datetime import datetime

HMessage="*High Alert*"
LMessage="Low Alert"
Mmessage="Medium Alert"
model = YOLO('./runs/detect/train6/weights/best.pt')
violence_count = 0
whatsapp_lock = threading.Lock()  

def sendMsg(message):
    with whatsapp_lock:  
        w.sendwhatmsg_instantly("+918188925203",message, 8, 38)
        pyautogui.click(1050, 950)
        time.sleep(3)
        k.press_and_release('enter')

def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        t -= 1
    return True

cap = cv2.VideoCapture('Studio_Project_V1.mp4')

while True:
    try:
        ret, frame = cap.read()
        name = None
        results = model.predict(source=frame, conf=0.4)

        if not results or len(results) == 0:
            continue
        else:
            no_fight_detected = True
            for result in results:
                annotator = Annotator(frame)
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
                    if name == "Violence":
                        violence_count += 1
                        print(violence_count, "Seconds")
                        start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                        cv2.putText(frame, "Timing " + str(start_time), (20, 650), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                                    (255, 0, 0), 1, cv2.LINE_AA, False)


                        if 170 <= violence_count <= 171:
                            print("Medium Alert: Alerting started")
                            end_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                            f = open("ReportFile.txt", "w+")
                            f.write(f"{Mmessage} Start timing of fight is {start_time}\n")
                            f.write(f"End timing of fight is {end_time}")
                            f.close()
                            mmsg = f"{Mmessage} Start timing of fight is {start_time}\n" + f" End timing of fight is {end_time}"
                            m = str(mmsg)
                            whatsappp = threading.Thread(target=sendMsg, args=(m,))
                            whatsappp.start()
                            print("Medium alert message sent")

                        elif 50 <= violence_count <= 51:
                            print("High Alert: Alerting started")
                            end_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                            f = open("ReportFile.txt", "w+")
                            f.write(f"{HMessage} Start timing of fight is {start_time}\n")
                            f.write(f" End timing of fight is {end_time}")
                            f.close()
                            Hmsg = f"{HMessage} Start timing of fight is {start_time}\n" + f" End timing of fight is {end_time}"
                            h = str(Hmsg)
                            whatsappp = threading.Thread(target=sendMsg, args=(h,))
                            whatsappp.start()
                            print("High alert message sent")
                            
                        elif 250 <= violence_count <= 251:
                            print("Low Alert: Alerting started")
                            end_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                            f = open("ReportFile.txt", "w+")
                            f.write(f"{LMessage} Start timing of fight is {start_time}\n")
                            f.write(f"End timing of fight is {end_time}" + "\n")
                            f.close()
                            lm = f"{LMessage} Start timing of fight is {start_time}\n" + f" End timing of fight is {end_time}"
                            l = str(lm)
                            whatsappp = threading.Thread(target=sendMsg, args=(l,))
                            whatsappp.start()
                            print("Low alert message sent")
        frame = annotator.result()
        frame = imutils.resize(frame, width=1200)
        cv2.imshow("detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        continue

cap.release()
cv2.destroyAllWindows()
