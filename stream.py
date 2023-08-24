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
from firebase_admin import db,storage,credentials
import firebase_admin

cred = credentials.Certificate("ai-yantra-firebase-adminsdk-cne24-b0a23517f9.json")
firebase_admin.initialize_app(
    cred, {"databaseURL": "https://ai-yantra-default-rtdb.firebaseio.com/",
           'storageBucket': 'gs://ai-yantra.appspot.com'}
)
ref = db.reference("/")

print(ref.get())

store = storage.bucket()
model = YOLO('./runs/detect/train6/weights/best.pt')
violence_count=0
def sendMsg(start_time,end_time):
        w.sendwhatmsg_instantly("+919755049483", f'violence Detected time {start_time} end time {end_time}', 8, 38)
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
cap = cv2.VideoCapture(2)
# cap = cv2.VideoCapture("rtsp://admin:MGINLP@192.168.1.35:554/cam/realmonitor?channel=1&subtype=0")   
while True:
    try:
        ret, frame = cap.read()
        name=None
        results = model.predict(source=frame,conf=0.4,save=True)
        if not results or len(results) == 0:
            continue
        else: 
            no_fight_detected = True
            for result in results:
                annotator = Annotator(frame)
                print(annotator)
                detection_count = result.boxes.shape[0]
                bd_boxes = result.boxes
                for box in bd_boxes:
                    b=box.xyxy[0]
                    c = box.cls
                    annotator.box_label(b,model.names[int(c)],color=(255,0,0), txt_color=(0,0,255))
                for i in range(detection_count):
                    cls = int(result.boxes.cls[i].item())
                    name = result.names[cls]
                    print(name)
                    if name == "Violence":
                        violence_count+=1
                        print(violence_count,"Seconds")
                        start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                        cv2.putText(frame, "Timing " + str(start_time), (20, 650), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                                    (255, 0, 0), 1, cv2.LINE_AA, False)
                    # if name=="NonViolence":
                    #     print("NonViolence detected")
                        # timer = threading.Thread(target=countdown, args=(2)) 
                        # count=timer.start()
                        # print(count, " its countdown seconds")
                        print("violence count",violence_count)
                        if violence_count==170 and violence_count<=180:
                                # timer = threading.Thread(target=countdown, args=(4)) 
                                # count=timer.start()
                                # if timer:
                                print("Alerting started")
                                end_time =datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                                f = open("ReportFile.txt","w+")
                                f.write(f"Start timing of fight is {start_time}")
                                f.write(f"End timing of fight is {end_time}"+ " Medium Violence")
                                f.close()
                                vviolence = 0
                                blob_name = f"violence{vviolence+1}.avi"
                                blob = store.blob(blob_name)
                                blob.upload_from_filename(f"PROJECT/Fight/{vviolence}.avi")
                                whatsappp = threading.Thread(target=sendMsg,args=(start_time,end_time))
                                whatsappp.start()
                                print("message send")
                                print("violence count",violence_count)
                        elif violence_count>=50 and violence_count<=60:
                            print("Alerting started")
                            end_time =datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                            f = open("ReportFile.txt","w+")
                            f.write(f"Start timing of fight is {start_time}")
                            f.write(f"End timing of fight is {end_time}"+ " High Alert Violence")
                            vviolence =0
                            blob_name = "image0.jpg"
                            blob = store.blob(blob_name)
                            blob.upload_from_filename("PROJECT/Fight/image0.jpg")

                            f.close()
                            whatsappp = threading.Thread(target=sendMsg,args=(start_time,end_time))
                            whatsappp.start()
                            print("message send")
                        elif violence_count>=250 and violence_count<=260:
                            print("Alerting started")
                            end_time =datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                            f = open("ReportFile.txt","w+")
                            f.write(f"Start timing of fight is {start_time}")
                            f.write(f"End timing of fight is {end_time}"+ " Low Violence")
                            f.close()
                            vviolence =0
                            blob_name = f"violence{vviolence+1}.avi"
                            blob = store.blob(blob_name)
                            blob.upload_from_filename(f"PROJECT/Fight/{vviolence}.avi")
                            whatsappp = threading.Thread(target=sendMsg,args=(start_time,end_time))
                            whatsappp.start()
                            print("message send")
                            print("violence count",violence_count)
            
        frame = annotator.result()
        # frame = imutils.resize(frame, width=480)
        frame = imutils.resize(frame,width=1200)
        # frame = cv2.flip(frame, 1)    
        cv2.imshow("detection",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        continue    
cap.release()
cv2.destroyAllWindows()

