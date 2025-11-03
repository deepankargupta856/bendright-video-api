import numpy as np
import math

# --- Helper Function to Calculate Angle ---
def calculate_angle(a, b, c):
    """
    Calculates the angle between three points (a, b, c).
    'b' is the vertex of the angle.
    Points are assumed to be landmark objects with .x and .y properties.
    """
    # Get the (x, y) coordinates
    p1 = np.array([a.x, a.y])  # First point
    p2 = np.array([b.x, b.y])  # Mid point (vertex)
    p3 = np.array([c.x, c.y])  # End point
    
    # Calculate radians
    radians = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - \
              np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    
    # Convert to degrees
    angle = np.abs(radians * 180.0 / math.pi)
    
    # Ensure angle is between 0 and 180
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# --- Main Scoring Engine Function ---
def check_warrior_2_pose(landmarks):
    """
    Analyzes the "Warrior II" pose using the provided landmarks.
    
    Args:
        landmarks: A dictionary of landmark objects (e.g., landmarks['LEFT_SHOULDER']).
    
    Returns:
        A tuple (score, messages) where 'score' is an int (0-100)
        and 'messages' is a list of feedback strings.
    """
    
    score = 100
    messages = []
    
    try:
        # --- 1. Check Arm Angles (Should be ~180 degrees) ---
        # We check from shoulder -> elbow -> wrist
        
        # Left Arm
        left_arm_angle = calculate_angle(
            landmarks['LEFT_SHOULDER'],
            landmarks['LEFT_ELBOW'],
            landmarks['LEFT_WRIST']
        )
        if left_arm_angle < 160:
            score -= 15
            messages.append("Straighten your left arm")

        # Right Arm
        right_arm_angle = calculate_angle(
            landmarks['RIGHT_SHOULDER'],
            landmarks['RIGHT_ELBOW'],
            landmarks['RIGHT_WRIST']
        )
        if right_arm_angle < 160:
            score -= 15
            messages.append("Straighten your right arm")

        # --- 2. Check Front Leg (Left Leg) Angle (Should be ~90 degrees) ---
        # We check from hip -> knee -> ankle
        left_knee_angle = calculate_angle(
            landmarks['LEFT_HIP'],
            landmarks['LEFT_KNEE'],
            landmarks['LEFT_ANKLE']
        )
        if left_knee_angle > 100 or left_knee_angle < 80:
            score -= 20
            messages.append("Bend your front (left) knee to 90 degrees")

        # --- 3. Check Back Leg (Right Leg) Angle (Should be ~180 degrees) ---
        right_knee_angle = calculate_angle(
            landmarks['RIGHT_HIP'],
            landmarks['RIGHT_KNEE'],
            landmarks['RIGHT_ANKLE']
        )
        if right_knee_angle < 160:
            score -= 20
            messages.append("Keep your back (right) leg straight")

        # --- 4. Check Torso (Should be upright) ---
        # A simple check: are the shoulders roughly aligned vertically with the hips?
        left_shoulder_x = landmarks['LEFT_SHOULDER'].x
        left_hip_x = landmarks['LEFT_HIP'].x
        
        # This threshold (0.1) may need tuning
        if abs(left_shoulder_x - left_hip_x) > 0.1: 
            score -= 15
            messages.append("Keep your torso upright")
            
        # --- Final Check ---
        if not messages:
            messages.append("Perfect Pose!")
        
        # Ensure score is not below 0
        if score < 0:
            score = 0
            
        return score, messages

    except (KeyError, AttributeError, TypeError):
        # This triggers if a required landmark (e.g., 'LEFT_SHOULDER') is not found
        # (e.g., person is out of frame)
        return 0, ["Full body not visible"]