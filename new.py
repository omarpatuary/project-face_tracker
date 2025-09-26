# By_SciCraft + Face Recognition Merge
import cv2
import numpy as np
import face_recognition
from cvzone.FaceDetectionModule import FaceDetector
import pyfirmata

# ------------------------------
# পরিচিত মুখ লোড করা
# ------------------------------

image1 = face_recognition.load_image_file("hasan.jpg")
encoding1 = face_recognition.face_encodings(image1)[0]


image2 = face_recognition.load_image_file("Omar.jpg")
encoding2 = face_recognition.face_encodings(image2)[0]

# image3 = face_recognition.load_image_file("tamjid.jpg")
# encoding3 = face_recognition.face_encodings(image3)[0]


image3 = face_recognition.load_image_file("প্রকৌশলী মোহাম্মদ আলাউদ্দিন.jpg")
encoding3 = face_recognition.face_encodings(image3)[0]


image4 = face_recognition.load_image_file("তানভীর আব্বাসী.jpg")
encoding4 = face_recognition.face_encodings(image4)[0]


# image6 = face_recognition.load_image_file("সোহেল মিয়া.jpg")
# encoding6 = face_recognition.face_encodings(image6)[0]



# image7 = face_recognition.load_image_file("আল মুবিন.jpg")
# encoding7 = face_recognition.face_encodings(image7)[0]



# image8 = face_recognition.load_image_file("মোহাম্মদ বিল্লাল হোসেন.jpg")
# encoding8 = face_recognition.face_encodings(image8)[0]



# image9 = face_recognition.load_image_file("সেবুল আহমদ.jpg")
# encoding9 = face_recognition.face_encodings(image9)[0]


# image10 = face_recognition.load_image_file("মাহবুব আলম শাকিল.jpg")
# encoding10 = face_recognition.face_encodings(image10)[0]



# image11 = face_recognition.load_image_file("মো.শিপন.jpg")
# encoding11 = face_recognition.face_encodings(image11)[0]



# image12 = face_recognition.load_image_file("মেহেদী হাসান.jpg")
# encoding12 = face_recognition.face_encodings(image12)[0]



# image13 = face_recognition.load_image_file("সোহেল মিয়া.jpg")
# encoding13 = face_recognition.face_encodings(image13)[0]



# image14 = face_recognition.load_image_file("সোহেল মিয়া.jpg")
# encoding14 = face_recognition.face_encodings(image14)[0]



known_faces = [encoding1, encoding2, encoding3, encoding4]
known_names = ["Hasan", "Omar", "Alauddin Sir", "Tanvir Sir"]

# ------------------------------
# ক্যামেরা সেটআপ
# ------------------------------
cap = cv2.VideoCapture(0)
ws, hs = 1280, 720
cap.set(3, ws)
cap.set(4, hs)

if not cap.isOpened():
    print("Camera couldn't Access!!!")
    exit()

# ------------------------------
# Arduino & Servo সেটআপ
# ------------------------------
port = "COM4"  # তোমার Arduino COM পোর্ট
board = pyfirmata.Arduino(port)
servo_pinX = board.get_pin('d:9:s')  # X-axis servo → pin 9
servo_pinY = board.get_pin('d:10:s') # Y-axis servo → pin 10

servoPos = [90, 90]  # initial servo position

# ------------------------------
# Face Detector
# ------------------------------
detector = FaceDetector()

# ------------------------------
# Main Loop
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --------------------------
    # Face Recognition
    # --------------------------
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_identified = False
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
            face_identified = True

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # --------------------------
    # Face Detection + Servo Tracking
    # --------------------------
    img, bboxs = detector.findFaces(frame, draw=False)
    if bboxs and face_identified:
        fx, fy = bboxs[0]["center"][0], bboxs[0]["center"][1]

        servoX = np.interp(fx, [0, ws], [180, 0])
        servoY = np.interp(fy, [0, hs], [180, 0])

        servoX = max(0, min(180, servoX))
        servoY = max(0, min(180, servoY))

        servoPos[0] = servoX
        servoPos[1] = servoY

        # লক্ষ্য চিহ্ন
        cv2.circle(frame, (fx, fy), 80, (0,0,255), 2)
        cv2.putText(frame, "TARGET LOCKED", (850,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    else:
        cv2.putText(frame, "NO TARGET", (880,50), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
        cv2.circle(frame, (640,360), 80, (0,0,255), 2)

    # --------------------------
    # Servo লিখা
    # --------------------------
    servo_pinX.write(servoPos[0])
    servo_pinY.write(servoPos[1])
    cv2.putText(frame, f'Servo X: {int(servoPos[0])} deg', (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),2)
    cv2.putText(frame, f'Servo Y: {int(servoPos[1])} deg', (50,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),2)

    # --------------------------
    # Display
    # --------------------------
    cv2.imshow("Face Identification & Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
