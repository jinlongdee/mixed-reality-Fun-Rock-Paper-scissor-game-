import cv2
import mediapipe as mp
import random
import time
import pygame 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# Initialize MediaPipe and Pygame
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)

pygame.mixer.init()
win_sound = pygame.mixer.Sound("sounds/win_sound.mp3")
lose_sound = pygame.mixer.Sound("sounds/lose_sound.mp3")
tie_sound = pygame.mixer.Sound("sounds/tie_sound.mp3")
victory_sound = pygame.mixer.Sound("sounds/victory_jingle.mp3")
countdown_sound = pygame.mixer.Sound("sounds/countdown.mp3")  # Add your 3-second sound file here
defeat = pygame.mixer.Sound("sounds/final_lose.mp3")
# Load gesture icons
gesture_icons = {
    "rock": cv2.imread("images/rock.png"),
    "paper": cv2.imread("images/paper.png"),
    "scissors": cv2.imread("images/scissors.png"),
    "ready": cv2.imread("images/ready.png")
}

# Function to overlay gesture icons
def overlay_icon(frame, icon, position, size=(200, 200)):
    icon_resized = cv2.resize(icon, size)  # Resize icon to specified size
    x, y = position
    h, w = size
    try:
        frame[y:y+h, x:x+w] = icon_resized  # Overlay the resized icon
    except ValueError:
        pass  # Handle out-of-bounds errors gracefully
    except KeyError:
        pass



# Function to display the scoreboard
def draw_scoreboard(frame, scores, rounds_to_candy):
    height, width, _ = frame.shape
    cv2.putText(frame, f"Rounds: {sum(scores.values())} / {rounds_to_candy}", (500, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    cv2.putText(frame, f"CPU Wins: {scores['cpu']}", (width - 300, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    cv2.putText(frame, f"Player Wins: {scores['player']}", (width - 1200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

# Function to display a bold title for the result
def display_result_title(frame, result):
    colors = {"win": (0, 255, 0), "lose": (0, 0, 255), "tie": (255, 255, 255)}
    messages = {"win": "YOU WIN!", "lose": "YOU LOSE!", "tie": "IT'S A TIE!"}
    color = colors.get(result, (255, 255, 255))
    message = messages.get(result, "")
    cv2.putText(frame, message, (frame.shape[1] // 2 - 300, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 5)

# Function to display instructions
def display_instructions(frame, text):
    cv2.putText(frame, text, (50, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

# Gesture detection function
def detect_gesture(landmarks):
    if landmarks:
        index_raised = landmarks[8].y < landmarks[6].y
        middle_raised = landmarks[12].y < landmarks[10].y
        ring_raised = landmarks[16].y < landmarks[14].y
        pinky_raised = landmarks[20].y < landmarks[18].y
        thumb_raised = landmarks[4].x < landmarks[2].x

        if not index_raised and not middle_raised and not ring_raised and not pinky_raised:
            return "rock"
        elif index_raised and middle_raised and ring_raised and pinky_raised:
            return "paper"
        elif index_raised and middle_raised and not ring_raised and not pinky_raised:
            return "scissors"
        elif index_raised and not any([middle_raised, ring_raised, pinky_raised]):
            return "ready"
    return None

# Game logic
def determine_winner(player, cpu):
    if player == cpu:
        return "tie"
    if (player == "rock" and cpu == "scissors") or \
       (player == "scissors" and cpu == "paper") or \
       (player == "paper" and cpu == "rock"):
        return "win"
    return "lose"

# Initialize webcam and game state
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
scores = {"player": 0, "cpu": 0, "tie": 0}
rounds_to_candy = 3

# Main game loop
while True:
    # Wait for the player to get ready
    ready = False
    ready_start_time = time.time()
    while not ready:
        ret, frame = cap.read()
        if not ret:
            continue
        #Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle around face
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray)
        
            if len(eyes) >= 1:
                left_eye = sorted(eyes, key=lambda e: e[0])[0]  # Select the leftmost eye
                ex, ey, ew, eh = left_eye
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = detect_gesture(hand_landmarks.landmark)
                if gesture == "ready":
                    if time.time() - ready_start_time > 0.5:
                        ready = True
                else:
                    ready_start_time = time.time()

        draw_scoreboard(frame, scores, rounds_to_candy)
        display_instructions(frame, "Raise your index finger to start!")
        cv2.imshow("Rock, Paper, Scissors", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Countdown with sound
    pygame.mixer.Sound.play(countdown_sound)
    for i in range(3, 0, -1):
        start_time = time.time()
        while time.time() - start_time < 1:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            draw_scoreboard(frame, scores, rounds_to_candy)
            cv2.putText(frame, f"{i}...", (frame.shape[1] // 2 - 50, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            cv2.imshow("Rock, Paper, Scissors", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                exit()

    # Player gesture detection
    player_choice = None
    start_time = time.time()
    while time.time() - start_time < 1:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = detect_gesture(hand_landmarks.landmark)
                if gesture:
                    player_choice = gesture
                    cv2.putText(frame, f"Gesture: {gesture.upper()}", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        #overlay_icon(frame, gesture_icons[player_choice], (50, 100), size=(200, 200))
        draw_scoreboard(frame, scores, rounds_to_candy)
        display_instructions(frame, "Show your move!")
        cv2.imshow("Rock, Paper, Scissors", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()
    
    # CPU makes a choice (hidden until now)
    cpu_choice = random.choice(["rock", "paper", "scissors"])
    result = determine_winner(player_choice, cpu_choice) if player_choice else "tie"

    # Display results
    for _ in range(60):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if player_choice:
            overlay_icon(frame, gesture_icons[player_choice], (50, 100))
        overlay_icon(frame, gesture_icons[cpu_choice], (frame.shape[1] - 250, 100), size=(200, 200))
        draw_scoreboard(frame, scores, rounds_to_candy)
        display_instructions(frame, f"Result: {result.upper()}!")
        cv2.imshow("Rock, Paper, Scissors", frame)
        cv2.waitKey(1)
    

    # Update scores and check victory
    if result == "win":
        scores["player"] += 1
        pygame.mixer.Sound.play(win_sound)
    elif result == "lose":
        scores["cpu"] += 1
        pygame.mixer.Sound.play(lose_sound)
    elif result == "tie":
        try:
            scores["tie"] += 1
        except KeyError:
            scores["tie"] = scores.get("tie", 0) + 1                
        pygame.mixer.Sound.play(tie_sound)

    if player_choice:
        result = determine_winner(player_choice, cpu_choice)
        for _ in range(30):  # Display result for 1 second (30 frames)
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            overlay_icon(frame, gesture_icons[player_choice], (50, 100), size=(200, 200))  # Player's choice
            overlay_icon(frame, gesture_icons[cpu_choice], (frame.shape[1] - 250, 100), size=(200, 200))  # CPU's choice
            draw_scoreboard(frame, scores, rounds_to_candy)
            display_result_title(frame, result)  # Show bold result title
            cv2.imshow("Rock, Paper, Scissors", frame)
            cv2.waitKey(1)
            
    if sum(scores.values()) == rounds_to_candy:
        if scores["player"] > scores["cpu"]:
            pygame.mixer.Sound.play(victory_sound)
            for _ in range(60):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, "You Win! Enjoy your candy!", (50, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                cv2.imshow("Rock, Paper, Scissors", frame)
                cv2.waitKey(1)
            scores = {"player": 0, "cpu": 0}
        else:
            pygame.mixer.Sound.play(defeat)
            for _ in range(60):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, "You Lose!!!", (50, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                cv2.imshow("Rock, Paper, Scissors", frame)
                cv2.waitKey(1)
            scores = {"player": 0, "cpu": 0}

    time.sleep(1)
