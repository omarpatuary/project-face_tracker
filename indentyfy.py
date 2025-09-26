import face_recognition
import cv2

# পরিচিত মুখ লোড করা
known_image = face_recognition.load_image_file("hasan.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# নাম লিস্ট
known_faces = [known_encoding]
known_names = ["hasan"]   # এখানে তুমি নিজের নাম দাও

# ক্যামেরা চালু
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ছোট করে প্রসেস করলে ফাস্ট হবে
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # মুখ খোঁজা
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # মিল খুঁজে বের করা
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # আসল ফ্রেমে স্কেল ঠিক করা
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # মুখে রেক্ট্যাঙ্গেল আঁকা
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Face Identification", frame)

    # 'q' চাপলে বের হয়ে যাবে
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
