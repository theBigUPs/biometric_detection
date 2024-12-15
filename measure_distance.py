#import cv2
import mediapipe as mp
import math

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

def mouth_width():
    '''Horizontal width of the mouth (distance between the corners of the mouth).'''
    #Left Mouth Corner	61
    #Right Mouth Corner	291
    pass

def mouth_to_chin():
    '''Vertical distance from the center of the mouth to the chin.'''
    pass

def face_width():
    '''Horizontal width of the face at the widest point near the cheekbones.'''
    pass

def face_height():
    '''Vertical distance from the top of the forehead (hairline) to the chin.'''
    pass

def ear_to_ear():
    '''Horizontal distance between the outer edges of both ears.'''
    pass

def eyebrow_to_eye():
    '''Vertical distance from the center of the eyebrows to the center of the eyes.'''
    pass

def eyebrow_len():
    '''Horizontal length of each eyebrow.'''
    pass

def forehead_height():
    '''Vertical distance from the hairline to the brow line.'''
    pass

def jawline_width():
    '''Distance between the angles of the mandible (jawbone).'''
    pass

