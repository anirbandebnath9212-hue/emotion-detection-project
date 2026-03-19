import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("emotion_model.h5")

# Face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Emotion labels
emotion_labels = ['Angry','Happy','Sad','Surprise','Neutral']

# Start camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 🔥 For stability
emotion_list = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        # 🔥 Improve quality
        face = cv2.equalizeHist(face)
        face = cv2.GaussianBlur(face, (3,3), 0)

        # Resize + normalize
        face = cv2.resize(face, (48,48))
        face = face / 255.0
        face = np.reshape(face, (1,48,48,1))

        # Predict (no spam logs)
        pred = model.predict(face, verbose=0)
        emotion = emotion_labels[np.argmax(pred)]

        # 🔥 Stabilize prediction
        emotion_list.append(emotion)
        if len(emotion_list) > 10:
            emotion_list.pop(0)

        emotion = max(set(emotion_list), key=emotion_list.count)

        # Draw box + text
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Emotion Detection", frame)

    # Exit with Q or ESC
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()