import os
import shutil
import uuid
from enum import Enum
from fastapi import FastAPI, UploadFile, File, Header, Form
from fastapi.responses import FileResponse, JSONResponse
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional
from starlette.background import BackgroundTask

# 1. UPDATED IMPORT: We now import the master analyzer function
from pose_analyzer import analyze_pose

# --- Initialize FastAPI App ---
app = FastAPI(title="Yoga Pose Analysis API")

# --- Initialize Mediapipe ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 2. NEW ENUM: Defines the selectable poses
class PoseName(str, Enum):
    warrior_2 = "warrior_2"
    tree_pose = "tree_pose"
    downward_dog = "downward_dog"

# --- Helper Function: The Core Video Processor (MODIFIED) ---
def process_video_frames(input_path: str, output_path: str, pose_name: str):
    """
    Reads the video from input_path, processes each frame for pose,
    draws landmarks, calculates score, and writes to output_path.
    
    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to save the annotated video.
        pose_name (str): The name of the pose to analyze (e.g., "warrior_2").
    
    Returns:
        (float) avg_score: The average score over all frames with a person.
        (str) final_message: The most common feedback message.
    """
    
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError("Error: Could not open video file.")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        cap.release()
        raise IOError("Error: Could not open VideoWriter for output.")

    frame_count = 0
    total_score = 0
    all_messages = []
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            current_score = 0
            current_messages = ["No person in frame"]
            
            if results.pose_landmarks:
                landmarks = {}
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    landmarks[mp_pose.PoseLandmark(i).name] = landmark
                
                # 3. MODIFIED LOGIC: Call the master function with the selected pose
                current_score, current_messages = analyze_pose(pose_name, landmarks)
                # -------------------------------

                total_score += current_score
                frame_count += 1
                all_messages.extend(current_messages)
                
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
            
            feedback_text = ", ".join(current_messages)
            
            cv2.putText(image, f'Score: {int(current_score)}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, feedback_text, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            out.write(image)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    avg_score = (total_score / frame_count) if frame_count > 0 else 0
    
    if not all_messages:
        final_message = "No person detected in video"
    else:
        final_message = max(set(all_messages), key=all_messages.count)
        
    return avg_score, final_message

# --- Cleanup Function (Unchanged) ---
def cleanup_temp_files(input_path: str, output_path: str):
    if os.path.exists(input_path):
        os.remove(input_path)
    if os.path.exists(output_path):
        os.remove(output_path)

# --- 4. The API Endpoint (MODIFIED) ---
@app.post("/analyze-pose/")
async def analyze_pose_endpoint(
    video: UploadFile = File(...),
    pose_selection: PoseName = Form(...)
):
    """
    Receives a video and a pose selection, processes it for correctness,
    and returns the annotated video.
    
    - **video**: The video file to analyze.
    - **pose_selection**: The name of the pose to analyze.
    """
    
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    input_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{video.filename}")
    output_path = os.path.join(temp_dir, f"{uuid.uuid4()}_output.mp4")

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # 5. MODIFIED CALL: Pass the pose_selection value to the processor
        score, message = process_video_frames(
            input_path, 
            output_path, 
            pose_name=pose_selection.value
        )

        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename="annotated_pose.mp4",
            headers={
                "X-Pose-Score": str(score),
                "X-Pose-Message": message
            },
            background=BackgroundTask(
                cleanup_temp_files, 
                input_path=input_path, 
                output_path=output_path
            )
        )

    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)
            
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred: {str(e)}"}
        )

@app.get("/")
def read_root():
    return {"message": "Welcome to the Yoga Pose Analysis API. Go to /docs to use it."}