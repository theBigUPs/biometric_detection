import cv2
import mediapipe as mp
import numpy as np

def load_image(path):
    image = cv2.imread(path)

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error: Could not load image from {path}")
        return None

    # Return the image object
    return image

def get_facial_features(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    landmarks_dict = {}

    if results.multi_face_landmarks:
        # Get the first face's landmarks (assuming one face)
        face_landmarks = results.multi_face_landmarks[0]

        # Store landmarks in a dictionary for easy access
        for i, landmark in enumerate(face_landmarks.landmark):
            landmarks_dict[i] = (landmark.x, landmark.y, landmark.z)

    return landmarks_dict


def check_tilt(landmarks_dict,image_width,image_height):#image
    model_points = np.array([
        (0.0, 0.0, 0.0),       # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye
        (225.0, 170.0, -135.0),   # Right eye
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ], dtype='double')

    # Map 2D image points to the 3D model points
    image_points = np.array([
        (landmarks_dict[1][0] * image_width, landmarks_dict[1][1] * image_height),   # Nose tip
        (landmarks_dict[152][0] * image_width, landmarks_dict[152][1] * image_height), # Chin
        (landmarks_dict[33][0] * image_width, landmarks_dict[33][1] * image_height),   # Left eye
        (landmarks_dict[263][0] * image_width, landmarks_dict[263][1] * image_height),  # Right eye
        (landmarks_dict[61][0] * image_width, landmarks_dict[61][1] * image_height),   # Left mouth corner
        (landmarks_dict[291][0] * image_width, landmarks_dict[291][1] * image_height)   # Right mouth corner
    ], dtype='double')

    # Camera intrinsic matrix (assuming no distortion)
    focal_length = image_width
    center = (image_width / 2, image_height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype='double'
    )

    # Assuming no lens distortion (if you have calibration, include distortion coefficients)
    dist_coeffs = np.zeros((4, 1))  # Distortion coefficients

    # Solve for the rotation and translation vectors using PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    # Get the rotation matrix from the rotation vector
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Calculate the Euler angles from the rotation matrix
    # Pitch (X-axis), Yaw (Y-axis), Roll (Z-axis)

    pitch = np.degrees(np.arctan2(rotation_matrix[2][1], rotation_matrix[2][2]))+180
    yaw = np.degrees(np.arctan2(-rotation_matrix[2][0], np.sqrt(rotation_matrix[2][1]**2 + rotation_matrix[2][2]**2)))
    roll = np.degrees(np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0]))

    # Return the Euler angles
    return {"pitch": pitch, "yaw": yaw, "roll": roll}


def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two 2D points (ignoring z-axis)."""
    return np.linalg.norm(np.array([point1[0], point1[1]]) - np.array([point2[0], point2[1]]))

def check_face_symmetry(landmarks_dict, image_width, image_height, tolerance=5):

    # Midline reference point: let's use the nose as the central reference
    nose_tip = landmarks_dict[1]  # Nose tip (landmark 1 in MediaPipe)
    
    # Calculate the centerline x-coordinate based on the nose (this will be used as reference)
    midline_x = nose_tip[0] * image_width
    
    # Define the indices for left and right landmarks (using MediaPipe landmark indices)
    symmetric_landmarks = {
        133: 362,   # Left eye (133) and right eye (362)
        70: 300,    # Left eyebrow (70) and right eyebrow (300)
        61: 291,    # Left mouth corner (61) and right mouth corner (291)
        234: 454,   # Left cheek (234) and right cheek (454)
        1: 1         # Nose tip (1) (this is already symmetrical)
    }
    
    # List to store asymmetry values
    symmetry_violations = []
    
    # Iterate through the landmark pairs and check their horizontal alignment
    for left_key, right_key in symmetric_landmarks.items():
        # Get the left and right points (each is a (x, y, z) tuple)
        left_point = landmarks_dict[left_key]
        right_point = landmarks_dict[right_key]
        
        # Convert normalized coordinates to pixel coordinates
        left_x = left_point[0] * image_width
        right_x = right_point[0] * image_width
        
        # Calculate the horizontal deviation from the midline for both points
        left_deviation = abs(left_x - midline_x)
        right_deviation = abs(right_x - midline_x)
        
        # Check if the horizontal deviations are within the tolerance
        if abs(left_deviation - right_deviation) > tolerance:
            #symmetry_violations.append(f"Asymmetry detected: {left_key} vs {right_key}")
            return False
    
    # Return True if no violations, otherwise return False and the violations
    # if symmetry_violations:
    #     return False
    # else:
    #     return True
    return True