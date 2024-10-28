import cv2
import mediapipe as mp
import numpy as np
import math
import os
import time
import tkinter as tk
from tkinter import ttk

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Drawing utilities to visualize the keypoints on the video
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points (A, B, C) where B is the vertex
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Second point (vertex)
    c = np.array(c)  # Third point

    # Calculate the angle
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

# Function to calculate speed of delivery based on distance moved per frame
def calculate_speed(prev_pos, curr_pos, frame_time):
    distance = math.sqrt((curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2)
    speed = distance / frame_time  # Speed in pixels per second
    return speed

# Function to normalize pose points for consistency (to adapt to different body sizes)
def normalize_landmarks(landmarks, image_shape):
    height, width = image_shape[:2]
    normalized_landmarks = []
    for lm in landmarks:
        normalized_landmarks.append([lm.x * width, lm.y * height])
    return normalized_landmarks

# Function to update Tkinter labels with current metrics
def update_gui(elbow_angle, shoulder_load, speed, suggestion, labels, chucking_status):
    labels['elbow_angle'].config(text=f'Elbow Angle: {int(elbow_angle)}°' if elbow_angle else "")
    labels['shoulder_load'].config(text=f'Shoulder Load: {int(shoulder_load)}°' if shoulder_load else "")
    labels['speed'].config(text=f'Speed: {speed:.2f} px/s' if speed else "")
    labels['suggestion'].config(text=f'Suggestion: {suggestion}' if suggestion != "N/A" else "")
    labels['chucking'].config(text=f'Chucking: {chucking_status}' if chucking_status != "N/A" else "")

# Function to process the video and calculate key metrics with feedback
def analyze_bowling_video(video_path, labels):
    cap = cv2.VideoCapture(video_path)

    # Variables to store previous frame time and previous ball position
    prev_frame_time = time.time()
    prev_ball_pos = None

    # History for long-term tracking (future scalability)
    elbow_angle_history = []
    speed_history = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video has ended or failed to load.")
            break

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose detection
        result = pose.process(image_rgb)

        elbow_angle = shoulder_load = speed = None
        suggestion = "N/A"
        chucking_status = "N/A"

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            # Normalize landmarks relative to frame size
            normalized_landmarks = normalize_landmarks(landmarks, frame.shape)

            # Get the shoulder, elbow, and wrist coordinates
            shoulder = normalized_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            elbow = normalized_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            wrist = normalized_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

            # Calculate the elbow angle
            elbow_angle = calculate_angle(shoulder, elbow, wrist)

            # Detect illegal chucking
            chucking_status = "Legal"
            if elbow_angle < 30:  # Adjust threshold based on legal elbow extension
                chucking_status = "Chucking (Illegal)"

            # Calculate shoulder load based on arm elevation
            shoulder_load = calculate_angle([0.5, 1.0], shoulder, elbow)

            # Suggest improvements based on elbow and shoulder angles
            if elbow_angle < 30:
                suggestion = "Consider elbow extension drills to improve your form."
            elif elbow_angle > 150:
                suggestion = "Optimize your arm action; elbow extension may be excessive."
            else:
                suggestion = "Elbow position is optimal. Maintain your form!"

            if shoulder_load > 80:
                suggestion += "\nInclude shoulder flexibility and strength training exercises."

            # Calculate speed using wrist movement as reference
            curr_ball_pos = wrist
            curr_frame_time = time.time()
            frame_time = curr_frame_time - prev_frame_time

            if prev_ball_pos is not None:
                speed = calculate_speed(prev_ball_pos, curr_ball_pos, frame_time)
            else:
                speed = 0

            # Save current metrics to history
            elbow_angle_history.append(elbow_angle)
            speed_history.append(speed)

            # Update previous ball position and frame time
            prev_ball_pos = curr_ball_pos
            prev_frame_time = curr_frame_time

        # Update Tkinter GUI with metrics and chucking detection
        update_gui(elbow_angle, shoulder_load, speed, suggestion, labels, chucking_status)

        # Display the frame with pose landmarks (for debugging purposes)
        mp_drawing.draw_landmarks(
            frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        cv2.imshow('Bowler Action Analysis', frame)

        # Break on keypress 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Tkinter UI for real-time data display
def create_tkinter_ui():
    window = tk.Tk()
    window.title("Bowler Action Analysis")

    # Set window size and layout
    window.geometry("400x350")

    # Set up the grid layout for labels
    ttk.Label(window, text="Bowler Action Analysis", font=("Helvetica", 16)).grid(row=0, column=0, columnspan=2, pady=(10, 5))

    elbow_angle_label = ttk.Label(window, text="", font=("Helvetica", 12))
    elbow_angle_label.grid(row=1, column=0, padx=10, pady=5)

    shoulder_load_label = ttk.Label(window, text="", font=("Helvetica", 12))
    shoulder_load_label.grid(row=2, column=0, padx=10, pady=5)

    speed_label = ttk.Label(window, text="", font=("Helvetica", 12))
    speed_label.grid(row=3, column=0, padx=10, pady=5)

    suggestion_label = ttk.Label(window, text="", font=("Helvetica", 12), wraplength=300)
    suggestion_label.grid(row=4, column=0, padx=10, pady=5)

    chucking_label = ttk.Label(window, text="", font=("Helvetica", 12))
    chucking_label.grid(row=5, column=0, padx=10, pady=5)

    # Store labels in a dictionary for easy access
    labels = {
        'elbow_angle': elbow_angle_label,
        'shoulder_load': shoulder_load_label,
        'speed': speed_label,
        'suggestion': suggestion_label,
        'chucking': chucking_label
    }

    return window, labels

# Main function to run the video analysis and Tkinter GUI
if __name__ == '__main__':
    # Path to your bowling video file
    video_path = r"D:\\FairPlay Unleashed AI for Legal Deliveries and Bowling Brilliance\\backend\\bowleractionanalysis.mp4"

    # Create Tkinter GUI
    window, labels = create_tkinter_ui()

    # Run video analysis in a separate thread or after the Tkinter window is ready
    window.after(100, lambda: analyze_bowling_video(video_path, labels))

    # Start Tkinter main loop
    window.mainloop()
