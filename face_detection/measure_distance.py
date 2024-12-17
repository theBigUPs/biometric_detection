#import cv2
#import mediapipe as mp
#import math
from face_detection import math

''' 
FEATURES NOT YET INCLUDED:

Golden Ratio: Proportions based on the "phi" ratio, such as the width-to-height ratio of the face.
Jawline Angle: Angle formed between the chin and the jawline on both sides.
Nose Width-to-Height Ratio: Ratio of the nose's width to its height.
Eye Aspect Ratio (EAR): Ratio of vertical and horizontal dimensions of the eye.
Lip Aspect Ratio (LAR): Ratio of vertical and horizontal dimensions of the lips.
'''

def interpupilary(landmark_dict):
    '''Distance between the centers of the pupils.'''


    # Extract the normalized coordinates of the key landmarks
    left_pupil = landmark_dict[468]# left iris landmark
    right_pupil = landmark_dict[473]# right iris landmark

    # Calculate Euclidean distance
    ipd_normalized = math.sqrt(
        (right_pupil[0] - left_pupil[0]) ** 2 + (right_pupil[1] - left_pupil[1]) ** 2
    )

    return ipd_normalized, # for non normalized value multiply by image width

def eye_to_eye(landmark_dict):
    '''Horizontal distance between the inner or outer corners of both eyes. 
       This implementation calculates the inner distance.'''

    left_inner_corner = landmark_dict[133]
    right_inner_corner = landmark_dict[362]

    # Calculate Euclidean distance using normalized coordinates
    eye_to_eye_normalized = math.sqrt(
        (right_inner_corner[0] - left_inner_corner[0]) ** 2 + (right_inner_corner[1] - left_inner_corner[1]) ** 2
    )

    return eye_to_eye_normalized

def eye_to_mouth(landmark_dict):
    '''Vertical distance from the center of the eyes to the center of the mouth.'''
    # Extract the normalized coordinates of the key landmarks
    left_pupil = landmark_dict[468]
    right_pupil = landmark_dict[473]
    mouth_center = landmark_dict[13]

    # Compute the center of the eyes (midpoint between left and right pupils)
    eye_center_y = (left_pupil[1] + right_pupil[1]) / 2
    mouth_y = mouth_center[1]

    # Calculate the vertical distance (difference in y-coordinates)
    vertical_distance_normalized = abs(mouth_y - eye_center_y)

    return vertical_distance_normalized

def eye_to_chin(landmark_dict):
    '''Vertical distance from the center of the eyes to the chin.'''
    left_pupil = landmark_dict[468]
    right_pupil = landmark_dict[473]
    chin_tip = landmark_dict[152]
    chin_tip_y=chin_tip[1]

    # Compute the center of the eyes (midpoint between left and right pupils)
    eye_center_y = (left_pupil[1] + right_pupil[1]) / 2
    vertical_distance_normalized = abs(chin_tip_y - eye_center_y)

    return vertical_distance_normalized

def nose_to_mouth(landmark_dict):
    '''Vertical distance from the base of the nose to the center of the mouth.'''
    mouth_center = landmark_dict[13]
    nose_base = landmark_dict[168]
    
    vertical_distance_normalized = abs(mouth_center[1] - nose_base[1])

    return vertical_distance_normalized


def nose_to_chin(landmark_dict):
    '''Vertical distance from the base of the nose to the tip of the chin.'''

    nose_base = landmark_dict[168]
    chin_tip = landmark_dict[152]
  
    vertical_distance_normalized = abs(chin_tip[1] - nose_base[1])

    return vertical_distance_normalized

def mouth_width(landmark_dict):
    '''Horizontal width of the mouth (distance between the corners of the mouth).'''
    #Left Mouth Corner	61
    #Right Mouth Corner	291
    left_mouth_corner = landmark_dict[61]
    right_mouth_corner = landmark_dict[291]

    horizontal_distance_normalized = abs(left_mouth_corner[0] - right_mouth_corner[0])

    return horizontal_distance_normalized

def mouth_to_chin(landmark_dict):
    '''Vertical distance from the center of the mouth to the chin.'''
    mouth_center = landmark_dict[13]
    chin_tip = landmark_dict[152]
    vertical_distance_normalized = abs(mouth_center[1] - chin_tip[1])

    return vertical_distance_normalized


def face_height(landmark_dict):
    '''Vertical distance from the top of the forehead (hairline) to the chin.'''
    forehead = landmark_dict[10]  # Forehead (near hairline) landmark
    chin_tip = landmark_dict[152]  # Chin landmark

    # Calculate the vertical distance (difference in y-coordinates)
    vertical_distance_normalized = abs(chin_tip[1] - forehead[1])

    return vertical_distance_normalized

def ear_to_ear(landmark_dict):
    '''Horizontal distance between the outer edges of both ears.'''
    left_ear = landmark_dict[234]  # Left ear outer edge landmark
    right_ear = landmark_dict[454]  # Right ear outer edge landmark

    # Calculate the horizontal distance (difference in x-coordinates)
    horizontal_distance_normalized = abs(right_ear[0] - left_ear[0])

    return horizontal_distance_normalized

def eyebrow_to_eye(landmark_dict):
    '''Vertical distance from the center of the eyebrows to the center of the eyes.'''
    left_eyebrow = landmark_dict[70]  # Left eyebrow center
    right_eyebrow = landmark_dict[300]  # Right eyebrow center
    left_eye = landmark_dict[468]  # Left eye center
    right_eye = landmark_dict[473]  # Right eye center

    # Calculate vertical distances (difference in y-coordinates)
    left_vertical_distance = abs(left_eye[1] - left_eyebrow[1])
    right_vertical_distance = abs(right_eye[1] - right_eyebrow[1])

    return left_vertical_distance, right_vertical_distance

def eyebrow_len(landmark_dict):
    '''Horizontal length of each eyebrow.'''
    left_eyebrow_start = landmark_dict[33]  # Left eyebrow start (inner)
    left_eyebrow_end = landmark_dict[133]  # Left eyebrow end (outer)
    
    right_eyebrow_start = landmark_dict[362]  # Right eyebrow start (inner)
    right_eyebrow_end = landmark_dict[263]  # Right eyebrow end (outer)

    # Calculate the horizontal distance (difference in x-coordinates) for each eyebrow
    left_eyebrow_length_normalized = abs(left_eyebrow_end[0] - left_eyebrow_start[0])
    right_eyebrow_length_normalized = abs(right_eyebrow_end[0] - right_eyebrow_start[0])

    return left_eyebrow_length_normalized, right_eyebrow_length_normalized

def forehead_height(landmark_dict):
    '''Vertical distance from the hairline to the brow line.'''
    hairline = landmark_dict[10]  # Hairline (near top of forehead)
    left_brow = landmark_dict[70]  # Left eyebrow (middle of the brow)
    right_brow = landmark_dict[300]  # Right eyebrow (middle of the brow)

    # Calculate the average y-coordinate of the brow line (average of left and right eyebrow)
    brow_line_y = (left_brow[1] + right_brow[1]) / 2

    # Calculate the vertical distance (difference in y-coordinates)
    vertical_distance_normalized = abs(brow_line_y - hairline[1])

    return vertical_distance_normalized

def jawline_width(landmark_dict):
    '''Distance between the angles of the mandible (jawbone).'''
    left_jaw_angle = landmark_dict[234]  # Left jaw angle (normalized)
    right_jaw_angle = landmark_dict[454]  # Right jaw angle (normalized)

    # Calculate the Euclidean distance between the left and right jaw angles
    distance_normalized = math.sqrt(
        (right_jaw_angle[0] - left_jaw_angle[0]) ** 2 + (right_jaw_angle[1] - left_jaw_angle[1]) ** 2
    )

    return distance_normalized

