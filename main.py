import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
face = mp_face.FaceMesh(min_detection_confidence=0.6)

images = {
    "normal": cv2.imread("assets/normal.jpg"),
    "idea": cv2.imread("assets/raise_finger.jpg"),
    "thinking": cv2.imread("assets/finger_on_mouth.jpg"),
    "smile": cv2.imread("assets/smile.jpg"),
}

def resize_img(img, height=480):
    h, w = img.shape[:2]
    new_w = int(w * (height / h))
    return cv2.resize(img, (new_w, height))

for k in images:
    images[k] = resize_img(images[k])

cap = cv2.VideoCapture(0)
current_state = "normal"

def count_raised_fingers(landmarks):
    fingers = []
    tips = [4, 8, 12, 16, 20]
    if landmarks.landmark[tips[0]].x < landmarks.landmark[tips[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for i in range(1, 5):
        if landmarks.landmark[tips[i]].y < landmarks.landmark[tips[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return sum(fingers)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face.process(rgb)
    hand_results = hands.process(rgb)
    current_state = "normal"

    if face_results.multi_face_landmarks:
        for fl in face_results.multi_face_landmarks:
            top_lip = fl.landmark[13].y * h
            bottom_lip = fl.landmark[14].y * h
            mouth_gap = abs(top_lip - bottom_lip)
            if mouth_gap > 15:
                current_state = "smile"

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            raised = count_raised_fingers(hand_landmarks)
            if raised >= 1:
                current_state = "idea"

            if face_results.multi_face_landmarks:
                mouth_center_y = face_results.multi_face_landmarks[0].landmark[13].y * h
                mouth_center_x = face_results.multi_face_landmarks[0].landmark[13].x * w
                index_tip = hand_landmarks.landmark[8]
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
                dist = np.sqrt((index_x - mouth_center_x)**2 + (index_y - mouth_center_y)**2)
                if dist < 50:
                    current_state = "thinking"

    box_w, box_h = int(w * 0.8), int(h * 0.8)
    x1 = (w - box_w) // 2
    y1 = (h - box_h) // 2
    x2, y2 = x1 + box_w, y1 + box_h
    color = (0, 255, 0) if current_state != "normal" else (255, 255, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label_map = {
        "normal": "Normal",
        "idea": "Idea",
        "thinking": "Thinking",
        "smile": "Smile",
    }
    label = label_map.get(current_state, "Normal")

    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(frame, (x1, y1 - 30), (x1 + t_size[0] + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    right_img = images.get(current_state, images["normal"])
    combined = np.hstack((frame, right_img))
    cv2.imshow("FacePoseTrigger", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break
    if cv2.getWindowProperty("FacePoseTrigger", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
