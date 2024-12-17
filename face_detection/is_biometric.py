from face_detection import np
from face_detection import mp
from face_detection import cv2

def check_background(image,threshold=80):
    std_dev = np.std(image)

    # Check if the standard deviation is below the threshold
    if std_dev < threshold:
        return True,std_dev  # Uniform background
    else:
        return False,std_dev  # Non-uniform background


def is_in_focus(image,threshold=100):
    '''Use Laplacian variance to measure focus (higher variance indicates better sharpness)'''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var > threshold 

def check_noise(image,threshold=10):
    '''Ensure the noise level is low. Use Gaussian blur to check for excessive noise'''
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    diff = cv2.absdiff(image, blurred)
    return diff.std() < threshold 


def enhance_lighting(image):
    ''' enhances lighting to make face detection easier'''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR) 

def check_face_area(image, min_percentage=30):
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

    # Convert the image to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)

    if results.detections:
        # Get the first face bounding box
        bboxC = results.detections[0].location_data.relative_bounding_box
        ih, iw, _ = image.shape
        
        # Convert relative bounding box to actual pixel values
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

        # Calculate the area of the face and the total image area
        face_area = w * h
        image_area = iw * ih
        face_percentage = (face_area / image_area) * 100

        #print(f"Face Area: {face_area}, Image Area: {image_area}, Percentage: {face_percentage}%")  # Debugging

        return face_percentage >= min_percentage,face_percentage
    #return False

def check_features(landmark_dict):

    '''
    checks if these landmarks are present in the landmarks dict 
    Left Eye	33 (outer), 133 (inner), 159/145 (center)
    Right Eye	263 (outer), 362 (inner), 386/374 (center)
    Nose Tip	1
    Left Nostril	98
    Right Nostril	327
    Upper Lip Center	13
    Lower Lip Center	14
    Left Mouth Corner	61
    Right Mouth Corner	291
    Chin Tip	152
    Left Jawline	234
    Right Jawline	454
    Left Ear	234
    Right Ear	454   
    Left Eyebrow	105 (outer), 66 (inner)
    Right Eyebrow	334 (outer), 296 (inner)
    Forehead Center	10
    Left Forehead Edge	127
    Right Forehead Edge	356  
    Left Iris  468
    Right Iris  473
    Nose Base 168
    Left Eyebrow 70
    Right Eyebrow 300
    left_eyebrow_start 33
    left_eyebrow_end 133
    right_eyebrow_start 362
    right_eyebrow_end 263
    '''
    landmark_list=[33,133,159,145,263,362,386,374,1,98,327
                   ,13,14,61,291,152,105,66,334,296,
                   10,127,356,234,454,468,473,168,70,300,33,133,362,263]
    for item in landmark_list:
        if item not in landmark_dict:
            return False


    return True


def calculate_inter_eye_distance(landmarks_dict, image_width, image_height, threshold):
    '''checks if the inter eye distance is at least 90 pixels as per spec'''
    # Left eye (133) and right eye (362)
    left_eye = landmarks_dict[133]
    right_eye = landmarks_dict[362]
    left_eye_pixel = (left_eye[0] * image_width, left_eye[1] * image_height)
    right_eye_pixel = (right_eye[0] * image_width, right_eye[1] * image_height)

    # Calculate Euclidean distance in pixels
    distance = ((left_eye_pixel[0] - right_eye_pixel[0])**2 +
                (left_eye_pixel[1] - right_eye_pixel[1])**2)**0.5

    return distance >= threshold, distance


def check_eye_open(landmarks_dict, image_width, image_height, eye='left', threshold=0.3):
    """
    Checks if the specified eye is open based on MediaPipe landmarks.
    """
    if eye == 'left':
        upper_lid = landmarks_dict[159]  # Upper eyelid
        lower_lid = landmarks_dict[145]  # Lower eyelid
        inner_corner = landmarks_dict[133]  # Inner corner
        outer_corner = landmarks_dict[33]  # Outer corner
    elif eye == 'right':
        upper_lid = landmarks_dict[386]  # Upper eyelid
        lower_lid = landmarks_dict[374]  # Lower eyelid
        inner_corner = landmarks_dict[362]  # Inner corner
        outer_corner = landmarks_dict[263]  # Outer corner
    else:
        raise ValueError("Invalid eye specified. Use 'left' or 'right'.")
    
    # Convert normalized coordinates to pixel coordinates
    upper_lid_y = upper_lid[1] * image_height
    lower_lid_y = lower_lid[1] * image_height
    inner_corner_x = inner_corner[0] * image_width
    outer_corner_x = outer_corner[0] * image_width

    # Calculate vertical distance between upper and lower eyelids
    eye_open_distance = abs(upper_lid_y - lower_lid_y)

    # Calculate horizontal eye width
    eye_width = abs(inner_corner_x - outer_corner_x)

    # Normalize the vertical distance by the eye width
    normalized_distance = eye_open_distance / eye_width

    # Check if the eye is open
    return normalized_distance > threshold




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
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,refine_landmarks=True)

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