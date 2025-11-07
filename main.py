import os
import shutil
import uuid
import base64
import asyncio
from enum import Enum
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware   # ✅ Import CORS middleware
import cv2
import mediapipe as mp
from pose_analyzer import analyze_pose

# --- Initialize FastAPI ---
app = FastAPI(title="Yoga Pose Analysis API")

# --- ✅ Add CORS configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 Allow all origins for now (you can restrict later)
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],  # allow all headers
)

# --- Initialize Mediapipe ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- In-memory store for processed videos ---
session_output_map = {}  # session_id -> output_path


# --- Enum for supported poses ---
class PoseName(str, Enum):
    warrior_2 = "warrior_2"
    tree_pose = "tree_pose"
    downward_dog = "downward_dog"


# --- Helper: Process video frames ---
def process_video_frames(input_path: str, output_path: str, pose_name: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video file.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        cap.release()
        raise IOError("Error: Could not open VideoWriter for output.")

    frame_count, total_score, all_messages = 0, 0, []

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

            current_score, current_messages = 0, ["No person in frame"]
            if results.pose_landmarks:
                landmarks = {
                    mp_pose.PoseLandmark(i).name: landmark
                    for i, landmark in enumerate(results.pose_landmarks.landmark)
                }

                current_score, current_messages = analyze_pose(pose_name, landmarks)
                total_score += current_score
                frame_count += 1
                all_messages.extend(current_messages)

                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                )

            cv2.putText(image, f"Score: {int(current_score)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            feedback_text = ", ".join(current_messages)
            cv2.putText(image, feedback_text[:80], (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            out.write(image)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    avg_score = (total_score / frame_count) if frame_count > 0 else 0
    final_message = (
        max(set(all_messages), key=all_messages.count)
        if all_messages else "No person detected in video"
    )
    return avg_score, final_message


# --- Cleanup Function ---
def cleanup_temp_files(input_path: str):
    if os.path.exists(input_path):
        os.remove(input_path)


# --- Upload & Process Endpoint ---
@app.post("/api/upload-video")
async def upload_and_process_video(
    video: UploadFile = File(...),
    pose: str = Form(...),
):
    """
    Receives a video and pose name from frontend,
    processes it, saves annotated video in temp folder,
    and returns metadata.
    """
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    unique_id = str(uuid.uuid4())
    input_path = os.path.join(temp_dir, f"{unique_id}_{video.filename}")
    output_path = os.path.join(temp_dir, f"{unique_id}_output.mp4")

    try:
        # Save uploaded video temporarily
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Process the video
        print(pose)
        score, message = process_video_frames(input_path, output_path, pose_name=pose)

        # Store mapping for WebSocket streaming later
        session_output_map[unique_id] = output_path

        # Return metadata
        return JSONResponse(
            status_code=200,
            content={
                "session_id": unique_id,
                "pose": pose,
                "score": score,
                "message": message,
            },
            background=BackgroundTask(cleanup_temp_files, input_path=input_path),
        )

    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)
        return JSONResponse(status_code=500, content={"error": str(e)})


# --- WebSocket Streaming Endpoint ---
@app.websocket("/ws/stream/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """
    Streams the processed video frames over WebSocket to the frontend.
    """
    await websocket.accept()
    print(f"🎥 WebSocket connected for session: {session_id}")

    try:
        if session_id not in session_output_map:
            await websocket.send_json({"status": "error", "message": "Invalid session_id"})
            await websocket.close()
            return

        video_path = session_output_map[session_id]
        if not os.path.exists(video_path):
            await websocket.send_json({"status": "error", "message": "Video file not found"})
            await websocket.close()
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            await websocket.send_json({"status": "error", "message": "Failed to open video"})
            await websocket.close()
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
        delay = 1.0 / fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        await websocket.send_json({"status": "progress", "message": "Starting stream..."})

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Encode each frame to JPEG -> base64
            _, buffer = cv2.imencode(".jpg", frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            await websocket.send_json({"status": "frame", "frame": frame_base64})

            current_frame += 1
            if current_frame % 30 == 0:
                progress_msg = f"Streaming... {int((current_frame / total_frames) * 100)}%"
                await websocket.send_json({"status": "progress", "message": progress_msg})

            await asyncio.sleep(delay)  # send frames at correct FPS

        await websocket.send_json({"status": "completed", "message": "Streaming completed!"})
        cap.release()
        await websocket.close()
        print(f"✅ Streaming finished for session {session_id}")

    except WebSocketDisconnect:
        print(f"⚠️ WebSocket disconnected for session {session_id}")
    except Exception as e:
        await websocket.send_json({"status": "error", "message": f"Server error: {str(e)}"})
        await websocket.close()


# --- Root Endpoint ---
@app.get("/")
def root():
    return {"message": "Yoga Pose Analysis API is running."}
