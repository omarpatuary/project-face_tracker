# By_SciCraft
import cv2
import face_recognition
from cvzone.FaceDetectionModule import FaceDetector
import pyfirmata
import numpy as np

face_locations = []
face_names = []


# ==========================
# Setup Known Face
# ==========================
known_image = face_recognition.load_image_file("hasan.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]
known_faces = [known_encoding]
known_names = ["hasan"]

# ==========================
# Setup Camera
# ==========================
cap = cv2.VideoCapture(0)
ws, hs = 640, 480  # smaller resolution for smoother fps
cap.set(3, ws)
cap.set(4, hs)

if not cap.isOpened():
    print("Camera couldn't access!")
    exit()

# ==========================
# Setup Arduino Servo
# ==========================
port = "COM4"
board = pyfirmata.Arduino(port)
servo_pinX = board.get_pin('d:9:s')
servo_pinY = board.get_pin('d:10:s')
servoPos = [90, 90]  # initial servo position

# ==========================
# Setup Face Detector
# ==========================
detector = FaceDetector()
frame_count = 0  # for skipping frames

# ==========================
# Main Loop
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Process every 2nd frame for speed
    if frame_count % 2 == 0:
        # ==========================
        # Face Recognition
        # ==========================
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]
            face_names.append(name)

        # ==========================
        # Face Detection for Servo
        # ==========================
        img, bboxs = detector.findFaces(frame, draw=False)

        if bboxs:
            fx, fy = bboxs[0]["center"]
            # convert coordinate to servo degree
            servoX = np.interp(fx, [0, ws], [180, 0])
            servoY = np.interp(fy, [0, hs], [180, 0])

            servoPos[0] = np.clip(servoX, 0, 180)
            servoPos[1] = np.clip(servoY, 0, 180)

            # Draw Target
            cv2.circle(frame, (fx, fy), 80, (0, 0, 255), 2)
            cv2.circle(frame, (fx, fy), 15, (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, "TARGET LOCKED", (ws-200, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
        else:
            cv2.putText(frame, "NO TARGET", (ws-200, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
            cv2.circle(frame, (ws//2, hs//2), 80, (0,0,255), 2)
            cv2.circle(frame, (ws//2, hs//2), 15, (0,0,255), cv2.FILLED)

        # Update Servo every 2nd frame
        servo_pinX.write(servoPos[0])
        servo_pinY.write(servoPos[1])

    # ==========================
    # Draw Face Recognition Boxes
    # ==========================
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Show Servo Position
    cv2.putText(frame, f'Servo X: {int(servoPos[0])} deg', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
    cv2.putText(frame, f'Servo Y: {int(servoPos[1])} deg', (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)

    cv2.imshow("Face & Servo Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
