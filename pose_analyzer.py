import numpy as np
import math

# --- Helper Function to Calculate Angle (Unchanged) ---
def calculate_angle(a, b, c):
    """
    Calculates the angle between three points (a, b, c).
    'b' is the vertex of the angle.
    Points are assumed to be landmark objects with .x and .y properties.
    """
    p1 = np.array([a.x, a.y])  # First point
    p2 = np.array([b.x, b.y])  # Mid point (vertex)
    p3 = np.array([c.x, c.y])  # End point
    
    radians = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - \
              np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    
    angle = np.abs(radians * 180.0 / math.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# --- 1. Warrior II Pose Logic (Unchanged) ---
def check_warrior_2_pose(landmarks):
    """Analyzes the 'Warrior II' pose."""
    score = 100
    messages = []
    
    try:
        # Left Arm
        left_arm_angle = calculate_angle(
            landmarks['LEFT_SHOULDER'], landmarks['LEFT_ELBOW'], landmarks['LEFT_WRIST']
        )
        if left_arm_angle < 160:
            score -= 15
            messages.append("Straighten your left arm")

        # Right Arm
        right_arm_angle = calculate_angle(
            landmarks['RIGHT_SHOULDER'], landmarks['RIGHT_ELBOW'], landmarks['RIGHT_WRIST']
        )
        if right_arm_angle < 160:
            score -= 15
            messages.append("Straighten your right arm")

        # Left Leg
        left_knee_angle = calculate_angle(
            landmarks['LEFT_HIP'], landmarks['LEFT_KNEE'], landmarks['LEFT_ANKLE']
        )
        if left_knee_angle > 100 or left_knee_angle < 80:
            score -= 20
            messages.append("Bend your front (left) knee to 90 degrees")

        # Right Leg
        right_knee_angle = calculate_angle(
            landmarks['RIGHT_HIP'], landmarks['RIGHT_KNEE'], landmarks['RIGHT_ANKLE']
        )
        if right_knee_angle < 160:
            score -= 20
            messages.append("Keep your back (right) leg straight")

        # Torso
        left_shoulder_x = landmarks['LEFT_SHOULDER'].x
        left_hip_x = landmarks['LEFT_HIP'].x
        if abs(left_shoulder_x - left_hip_x) > 0.1: 
            score -= 15
            messages.append("Keep your torso upright")
            
        if not messages:
            messages.append("Perfect Pose!")
        
        if score < 0:
            score = 0
            
        return score, messages

    except (KeyError, AttributeError, TypeError):
        return 0, ["Full body not visible"]

# --- 2. Tree Pose Logic (NEWLY ADDED) ---
def check_tree_pose(landmarks):
    """
    Analyzes the 'Tree' pose.
    Assumes standing on the LEFT leg, with RIGHT foot on the thigh.
    """
    score = 100
    messages = []
    
    try:
        # --- 1. Check Standing Leg (Left) ---
        # Angle from hip -> knee -> ankle should be ~180 degrees.
        standing_leg_angle = calculate_angle(
            landmarks['LEFT_HIP'], landmarks['LEFT_KNEE'], landmarks['LEFT_ANKLE']
        )
        if standing_leg_angle < 170:
            score -= 30
            messages.append("Straighten your standing (left) leg")

        # --- 2. Check Bent Leg (Right) ---
        # Angle from hip -> knee -> ankle should be sharply bent.
        bent_leg_angle = calculate_angle(
            landmarks['RIGHT_HIP'], landmarks['RIGHT_KNEE'], landmarks['RIGHT_ANKLE']
        )
        if bent_leg_angle > 90: # We expect a sharp bend
            score -= 30
            messages.append("Bring your right foot up to your thigh")

        # --- 3. Check Bent Knee Position ---
        # The bent knee should point outwards, not forwards.
        # We check the angle Shoulder -> Hip -> Knee
        bent_hip_angle = calculate_angle(
            landmarks['RIGHT_SHOULDER'], landmarks['RIGHT_HIP'], landmarks['RIGHT_KNEE']
        )
        if bent_hip_angle < 90: # Expecting an open hip angle
            score -= 20
            messages.append("Open your right hip by pointing your knee outwards")

        # --- 4. Check Torso (Upright) ---
        # Check vertical alignment of shoulder and hip
        left_shoulder_x = landmarks['LEFT_SHOULDER'].x
        left_hip_x = landmarks['LEFT_HIP'].x
        if abs(left_shoulder_x - left_hip_x) > 0.1: 
            score -= 20
            messages.append("Keep your torso upright and balanced")

        if not messages:
            messages.append("Great balance!")
        
        if score < 0:
            score = 0
            
        return score, messages

    except (KeyError, AttributeError, TypeError):
        return 0, ["Full body not visible"]

# --- 3. Downward Dog Logic (NEWLY ADDED) ---
def check_downward_dog_pose(landmarks):
    """
    Analyzes the 'Downward Dog' pose.
    Forms an inverted 'V' shape.
    """
    score = 100
    messages = []
    
    try:
        # --- 1. Check Arms & Back Line ---
        # Angle from Wrist -> Shoulder -> Hip should be ~180 degrees (a straight line).
        right_arm_back_angle = calculate_angle(
            landmarks['RIGHT_WRIST'], landmarks['RIGHT_SHOULDER'], landmarks['RIGHT_HIP']
        )
        left_arm_back_angle = calculate_angle(
            landmarks['LEFT_WRIST'], landmarks['LEFT_SHOULDER'], landmarks['LEFT_HIP']
        )
        
        if (right_arm_back_angle < 160) or (left_arm_back_angle < 160):
            score -= 30
            messages.append("Straighten your arms and back; press hips up")

        # --- 2. Check Leg Straightness ---
        # Angle from Hip -> Knee -> Ankle should be ~180 degrees.
        right_leg_angle = calculate_angle(
            landmarks['RIGHT_HIP'], landmarks['RIGHT_KNEE'], landmarks['RIGHT_ANKLE']
        )
        left_leg_angle = calculate_angle(
            landmarks['LEFT_HIP'], landmarks['LEFT_KNEE'], landmarks['LEFT_ANKLE']
        )
        
        if (right_leg_angle < 160) or (left_leg_angle < 160):
            score -= 30
            messages.append("Straighten your legs (it's okay to bend if hamstrings are tight)")

        # --- 3. Check Body 'V' Shape ---
        # Angle at the hips (Shoulder -> Hip -> Knee) should be an angle, not flat.
        right_hip_angle = calculate_angle(
            landmarks['RIGHT_SHOULDER'], landmarks['RIGHT_HIP'], landmarks['RIGHT_KNEE']
        )
        left_hip_angle = calculate_angle(
            landmarks['LEFT_SHOULDER'], landmarks['LEFT_HIP'], landmarks['LEFT_KNEE']
        )
        
        if (right_hip_angle < 60) or (left_hip_angle < 60):
            score -= 20
            messages.append("Lift your hips higher to form an inverted 'V'")
        
        if (right_hip_angle > 120) or (left_hip_angle > 120):
            score -= 20
            messages.append("Bring your body into an inverted 'V', don't be too flat")


        if not messages:
            messages.append("Excellent Downward Dog!")
        
        if score < 0:
            score = 0
            
        return score, messages

    except (KeyError, AttributeError, TypeError):
        return 0, ["Full body not visible"]

# --- 4. Master Pose Analyzer (Unchanged) ---
def analyze_pose(pose_name: str, landmarks):
    """
    Dispatcher function to call the correct pose analysis.
    """
    if pose_name == "warrior-ii":
        return check_warrior_2_pose(landmarks)
    elif pose_name == "tree-pose":
        return check_tree_pose(landmarks)
    elif pose_name == "downward-dog":
        return check_downward_dog_pose(landmarks)
    else:
        return 0, ["Selected pose is not supported"]