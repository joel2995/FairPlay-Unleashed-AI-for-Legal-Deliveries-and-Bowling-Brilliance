import cv2
import mediapipe as mp
import numpy as np
import math
import os
import time
import json

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

# Function to process the video and calculate key metrics with feedback
def analyze_bowling_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    
    # Variables to store previous frame time and previous ball position
    prev_frame_time = time.time()
    prev_ball_pos = None

    # History for long-term tracking
    elbow_angle_history = []
    speed_history = []
    frame_results = []
    
    # Create output directory for frames
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    frame_count = 0
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(output_dir, "analyzed_video.mp4")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("Starting video analysis...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose detection
        result = pose.process(image_rgb)

        elbow_angle = shoulder_load = speed = None
        suggestion = "N/A"
        chucking_status = "N/A"
        frame_data = {}

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
            
            # Store frame data
            frame_data = {
                "frame_number": frame_count,
                "elbow_angle": float(elbow_angle) if elbow_angle is not None else None,
                "shoulder_load": float(shoulder_load) if shoulder_load is not None else None,
                "speed": float(speed) if speed is not None else 0,
                "suggestion": suggestion,
                "chucking_status": chucking_status
            }

        # Draw landmarks on the frame
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
            
            # Add text annotations to the frame
            if elbow_angle is not None:
                cv2.putText(frame, f"Elbow: {int(elbow_angle)}°", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if shoulder_load is not None:
                cv2.putText(frame, f"Shoulder: {int(shoulder_load)}°", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if speed is not None:
                cv2.putText(frame, f"Speed: {speed:.2f} px/s", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Status: {chucking_status}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if chucking_status == "Chucking (Illegal)" else (0, 255, 0), 2)
        
        # Save frame with annotations
        if frame_data:
            frame_filename = f"frame_{frame_count:04d}.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            
            # Add frame path to data
            frame_data["frame_path"] = os.path.join("frames", frame_filename)
            frame_results.append(frame_data)
        
        # Write frame to output video
        out.write(frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video analysis completed!")
    
    # Calculate summary statistics
    if elbow_angle_history:
        avg_elbow_angle = sum(elbow_angle_history) / len(elbow_angle_history)
        min_elbow_angle = min(elbow_angle_history)
        max_elbow_angle = max(elbow_angle_history)
    else:
        avg_elbow_angle = min_elbow_angle = max_elbow_angle = None
    
    if speed_history:
        avg_speed = sum(speed_history) / len(speed_history)
        max_speed = max(speed_history)
    else:
        avg_speed = max_speed = None
    
    # Determine overall chucking status
    overall_chucking = "Legal"
    for frame in frame_results:
        if frame.get("chucking_status") == "Chucking (Illegal)":
            overall_chucking = "Chucking (Illegal)"
            break
    
    # Create summary
    summary = {
        "total_frames": frame_count,
        "avg_elbow_angle": float(avg_elbow_angle) if avg_elbow_angle is not None else None,
        "min_elbow_angle": float(min_elbow_angle) if min_elbow_angle is not None else None,
        "max_elbow_angle": float(max_elbow_angle) if max_elbow_angle is not None else None,
        "avg_speed": float(avg_speed) if avg_speed is not None else None,
        "max_speed": float(max_speed) if max_speed is not None else None,
        "overall_chucking_status": overall_chucking,
        "overall_suggestion": suggestion,
        "output_video_path": output_video_path
    }
    
    # Create final results
    results = {
        "frames": frame_results,
        "summary": summary
    }
    
    # Save results to JSON file
    results_path = os.path.join(output_dir, "analysis_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {results_path}")
    print(f"Analyzed video saved to {output_video_path}")
    
    return results

# Main function to run the video analysis
if __name__ == '__main__':
    # Path to your bowling video file
    video_path = r"D:\FairPlay Unleashed AI for Legal Deliveries and Bowling Brilliance\backend\bowleractionanalysis.mp4"
    
    # Create output directory
    output_dir = r"D:\FairPlay Unleashed AI for Legal Deliveries and Bowling Brilliance\uploads\analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the analysis
    results = analyze_bowling_video(video_path, output_dir)
    
    # Print summary results
    print("\nAnalysis Summary:")
    print(f"Total Frames: {results['summary']['total_frames']}")
    print(f"Average Elbow Angle: {results['summary']['avg_elbow_angle']:.2f}°")
    print(f"Min Elbow Angle: {results['summary']['min_elbow_angle']:.2f}°")
    print(f"Max Elbow Angle: {results['summary']['max_elbow_angle']:.2f}°")
    print(f"Average Speed: {results['summary']['avg_speed']:.2f} px/s")
    print(f"Max Speed: {results['summary']['max_speed']:.2f} px/s")
    print(f"Overall Chucking Status: {results['summary']['overall_chucking_status']}")
    print(f"Overall Suggestion: {results['summary']['overall_suggestion']}")
    print(f"\nAnalyzed video saved to: {results['summary']['output_video_path']}")
