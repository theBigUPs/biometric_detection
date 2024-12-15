
import is_biometric as ib
import measure_distance as md
#import cv2
#import mediapipe as mp
'''
# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Read an image
image = cv2.imread("test.PNG")
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image
results = face_mesh.process(rgb_image)

# Draw landmarks
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for id, landmark in enumerate(face_landmarks.landmark):
            h, w, c = image.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (x, y), 1, (255, 0, 0), -1)

cv2.imshow("Landmarks", image)
cv2.waitKey(0)'''
'''
import cv2
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Read an image
image = cv2.imread("tt2.jpg")
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image
results = face_mesh.process(rgb_image)

# Define the ear landmark indices
LEFT_EAR_LANDMARKS = [234]#, 236, 218

# Right ear (reduced landmarks)
RIGHT_EAR_LANDMARKS = [454]#, 447, 437

# Draw only the ear landmarks
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for id, landmark in enumerate(face_landmarks.landmark):
            if id in LEFT_EAR_LANDMARKS or id in RIGHT_EAR_LANDMARKS:
                h, w, c = image.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Green dots for the ears

# Show the resulting image
cv2.imshow("Ear Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()'''


"""

mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB (MediaPipe uses RGB)

    # Initialize Face Detection model
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        # Perform face detection
        results = face_detection.process(image_rgb)

        # If faces are detected, draw bounding boxes
        if results.detections:
            for detection in results.detections:
                # Draw bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Optionally, draw the landmarks (optional)
                mp_drawing.draw_detection(image, detection)

    # Show the image with bounding boxes
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""


def main():
    image = ib.load_image("tt6.jpg")
    height, width, channels = image.shape
    
    img_dict = ib.get_facial_features(image)
    #bigger_than_90, dist = ib.calculate_inter_eye_distance(img_dict,width,height,90)
    #left_eye = ib.check_eye_open(img_dict,width,height,"left")
    #right_eye = ib.check_eye_open(img_dict,width,height,"right")
    #print(img_dict)
    #tilt = check.check_tilt(img_dict, image_height=height, image_width=width)
    #print(tilt)
    #print(height)
    #print(width)
    #sym = check.check_face_symmetry(img_dict,width,height,5)
    #print(left_eye)
    #print(right_eye)
    #features_present = ib.check_features(img_dict)
    
    #print(dist)
    #print(features_present)
    #lighting_adjust = ib.enhance_lighting(image)
    #image = lighting_adjust
    #back,dev = ib.check_background(image)
    #focus = ib.is_in_focus(image) 
    #noise = ib.check_noise(image)
    
    
    
    #face_area,ratio = ib.check_face_area(image,70)
    #print(f"features:{features_present}     res:{bigger_than_90}     area:{face_area}      ratio:{ratio}")
    #features_present = ib.check_features(img_dict)
    #print(f"back:{back}   backdev:{dev}  focus:{focus}   noise:{noise}   features:{features_present}")
    #print(img_dict[168])
    #inter = md.interpupilary(img_dict)
    #print(inter)


if __name__ == "__main__":
    main()