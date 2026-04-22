import cv2
import mediapipe as mp
import math
import sys

# --- Helper Functions ---

def get_pixel_distance(lm1, lm2, img_w, img_h):
    """Euclidean distance in pixel space between two normalized MediaPipe landmarks."""
    x1, y1 = int(lm1.x * img_w), int(lm1.y * img_h)
    x2, y2 = int(lm2.x * img_w), int(lm2.y * img_h)
    return math.hypot(x1 - x2, y1 - y2)

def check_hand_on_head(hand_landmarks, face_landmarks, w, h):
    """
    Checks if a hand's fingertips are near the temples and above the eyebrows.
    This function is called *after* we've confirmed face_landmarks exists.
    """
    # Use face landmarks for left/right temple and eyebrows
    head_side_left = face_landmarks[234]
    head_side_right = face_landmarks[454]
    eyebrow_left = face_landmarks[105]
    eyebrow_right = face_landmarks[334]
    
    avg_eyebrow_y = (eyebrow_left.y + eyebrow_right.y) / 2.0

    # check index and middle fingertips
    for idx in (8, 12): # Index and Middle tips
        tip = hand_landmarks.landmark[idx]
        
        # compare to both temples (pixel distance)
        d_left = get_pixel_distance(tip, head_side_left, w, h)
        d_right = get_pixel_distance(tip, head_side_right, w, h)
        
        # If near either temple AND above the eyebrows (smaller 'y' is higher)
        if (d_left < HEAD_TOUCH_PIXEL_THRESHOLD or d_right < HEAD_TOUCH_PIXEL_THRESHOLD) and \
           (tip.y < avg_eyebrow_y):
            return True
            
    return False # No fingertips were on the head

# --- Constants and Thresholds (Tune these!) ---
SMILE_NORM_THRESHOLD = 0.65     # normalized mouth/eye ratio
THINKING_PIXEL_THRESHOLD = 40    # ~pixels; distance from fingertip to mouth
HEAD_TOUCH_PIXEL_THRESHOLD = 60  # ~pixels; fingertip near temple

# --- Initialization ---
print("Loading models and images...")

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Create objects
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load meme images (update your path if needed)
assets_path = "assets/"

def safe_imread(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img

try:
    meme_normal = safe_imread(assets_path + 'monkey_normal.jpg')
    meme_smile = safe_imread(assets_path + 'monkey_smile.jpg')
    meme_idea = safe_imread(assets_path + 'monkey_idea.jpg')
    meme_thinking = safe_imread(assets_path + 'monkey_thinking.jpg')
    meme_mind_blown = safe_imread(assets_path + 'think_monkey_think.jpg')  # new image

    MEME_SIZE = (400, 400)
    meme_normal = cv2.resize(meme_normal, MEME_SIZE)
    meme_smile = cv2.resize(meme_smile, MEME_SIZE)
    meme_idea = cv2.resize(meme_idea, MEME_SIZE)
    meme_thinking = cv2.resize(meme_thinking, MEME_SIZE)
    meme_mind_blown = cv2.resize(meme_mind_blown, MEME_SIZE)
except Exception as e:
    print("Error loading images:", e)
    print("Make sure all meme .jpg files exist in:", assets_path)
    hands.close()
    face_mesh.close()
    sys.exit(1)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    hands.close()
    face_mesh.close()
    sys.exit(1)

print("Starting Monkey Meme Detector... Press 'q' to quit.")

# --- Main Loop ---
try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        # mark frame as not writeable for performance
        frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = hands.process(rgb_frame)
        face_results = face_mesh.process(rgb_frame)

        # mark frame writeable again
        frame.flags.writeable = True

        gesture = "Normal"
        meme_to_show = meme_normal
        h, w, _ = frame.shape

        face_landmarks = None
        if face_results.multi_face_landmarks:
            # take first face
            face_landmarks = face_results.multi_face_landmarks[0].landmark

        # --- 1) MIND BLOWN: require face AND exactly two hands ---
        if face_landmarks and hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) == 2:
            hand_landmarks_list = hand_results.multi_hand_landmarks
            
            # Call the helper function (now defined at the top) for each hand
            hand1_on_head = check_hand_on_head(hand_landmarks_list[0], face_landmarks, w, h)
            hand2_on_head = check_hand_on_head(hand_landmarks_list[1], face_landmarks, w, h)

            if hand1_on_head and hand2_on_head:
                gesture = "MIND BLOWN!"
                meme_to_show = meme_mind_blown

       # --- 2) THINKING: index finger near mouth corners --- (MOVED UP)
        if gesture == "Normal" and face_landmarks and hand_results.multi_hand_landmarks:
            mouth_left = face_landmarks[61]
            mouth_right = face_landmarks[291]

            for hand_landmarks in hand_results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                dist_left = get_pixel_distance(index_tip, mouth_left, w, h)
                dist_right = get_pixel_distance(index_tip, mouth_right, w, h)
                if dist_left < THINKING_PIXEL_THRESHOLD or dist_right < THINKING_PIXEL_THRESHOLD:
                    gesture = "Thinking"
                    meme_to_show = meme_thinking
                    break # Found gesture, stop checking hands

        # --- 3) IDEA: one finger up (index up, middle folded) --- (MOVED DOWN)
        if gesture == "Normal" and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Check if this hand already triggered 'Thinking'
                if gesture != "Normal":
                    break 

                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                is_index_raised = (index_tip.y < index_pip.y)
                is_middle_folded = (middle_tip.y > middle_pip.y)
                is_hand_up = (index_tip.y < wrist.y)

                if is_index_raised and is_middle_folded and is_hand_up:
                    gesture = "Idea"
                    meme_to_show = meme_idea
                    break # Found gesture, stop checking hands
                
        # --- 4) SMILE: measure normalized mouth / eye distance ---
        if gesture == "Normal" and face_landmarks:
            mouth_left = face_landmarks[61]
            mouth_right = face_landmarks[291]
            eye_left = face_landmarks[33]
            eye_right = face_landmarks[263]

            # Use normalized coordinates for a scale-invariant ratio
            smile_dist = math.hypot((mouth_left.x - mouth_right.x), (mouth_left.y - mouth_right.y))
            eye_dist = math.hypot((eye_left.x - eye_right.x), (eye_left.y - eye_right.y))
            normalized_smile = smile_dist / (eye_dist + 1e-6) # add 1e-6 to avoid division by zero

            if normalized_smile > SMILE_NORM_THRESHOLD:
                gesture = "Smile"
                meme_to_show = meme_smile

        # --- Draw landmarks (optional) ---
        if face_results.multi_face_landmarks:
            for face_lms in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_lms,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1)
                )

        if hand_results.multi_hand_landmarks:
            for hand_lms in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_lms,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                )

        # --- Display UI Elements ---
        cv2.putText(frame, gesture, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('Monkey Meme Detector', frame)
        cv2.imshow('Meme Reaction', meme_to_show)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

finally:
    # This 'finally' block ensures cleanup happens even if an error occurs in the loop
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    face_mesh.close()
    