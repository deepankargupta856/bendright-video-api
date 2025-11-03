import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, Header
from fastapi.responses import FileResponse, JSONResponse
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional

# 1. NEW IMPORT
from starlette.background import BackgroundTask

# Import our pose analysis functions
from pose_analyzer import check_warrior_2_pose

# --- Initialize FastAPI App ---
app = FastAPI(title="Yoga Pose Analysis API")

# --- Initialize Mediapipe ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Helper Function: The Core Video Processor ---
def process_video_frames(input_path: str, output_path: str):
    """
    Reads the video from input_path, processes each frame for pose,
    draws landmarks, calculates score, and writes to output_path.
    
    Returns:
        (float) avg_score: The average score over all frames with a person.
        (str) final_message: The most common feedback message.
    """
    
    cap = cv2.VideoCapture(input_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        raise ValueError("Error: Could not open video file.")
    
    # Get video properties for VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 'mp4v' is a good codec for .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Check if VideoWriter was created successfully
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
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
            
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            current_score = 0
            current_messages = ["No person in frame"]
            
            # --- Pose Logic ---
            if results.pose_landmarks:
                # Create the landmark dictionary
                landmarks = {}
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    landmarks[mp_pose.PoseLandmark(i).name] = landmark
                
                # --- Call our Scoring Engine ---
                current_score, current_messages = check_warrior_2_pose(landmarks)
                # -------------------------------

                total_score += current_score
                frame_count += 1 # Only count frames where a person is detected
                all_messages.extend(current_messages) # Collect all messages
                
                # Draw the pose landmarks
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
            
            # --- Put Score & Feedback on Frame ---
            feedback_text = ", ".join(current_messages)
            
            cv2.putText(image, f'Score: {int(current_score)}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, feedback_text, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # Write the annotated frame
            out.write(image)

    # --- Cleanup ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # --- Calculate Final Score & Message ---
    avg_score = (total_score / frame_count) if frame_count > 0 else 0
    
    if not all_messages:
        final_message = "No person detected in video"
    else:
        # Find the most common feedback message
        final_message = max(set(all_messages), key=all_messages.count)
        
    return avg_score, final_message

# 2. NEW HELPER FUNCTION FOR CLEANUP
def cleanup_temp_files(input_path: str, output_path: str):
    """
    Removes the temporary input and output files.
    This will be run in the background after the response is sent.
    """
    if os.path.exists(input_path):
        os.remove(input_path)
    if os.path.exists(output_path):
        os.remove(output_path)

# --- The API Endpoint (MODIFIED) ---
@app.post("/analyze-pose/")
async def analyze_pose_endpoint(video: UploadFile = File(...)):
    """
    Receives a video, processes it for pose correctness,
    and returns the annotated video.
    
    The correctness score and feedback message are returned
    in custom HTTP headers: `X-Pose-Score` and `X-Pose-Message`.
    """
    
    # Create a temporary directory
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create unique temp file paths
    input_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{video.filename}")
    output_path = os.path.join(temp_dir, f"{uuid.uuid4()}_output.mp4")

    try:
        # 1. Save uploaded video to temp input path
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # 2. Process the video (This is the heavy lifting)
        score, message = process_video_frames(input_path, output_path)

        # 3. Return the processed video file
        # We add the cleanup task to the 'background' argument.
        # This will run AFTER the file is sent.
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
        # If processing fails, we must still clean up the input file
        # (the output file likely was never created)
        if os.path.exists(input_path):
            os.remove(input_path)
            
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred: {str(e)}"}
        )
    # 4. THE 'finally' BLOCK IS GONE!

@app.get("/")
def read_root():
    return {"message": "Welcome to the Yoga Pose Analysis API. Go to /docs to use it."}