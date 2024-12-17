
from deepface import DeepFace

# Paths to the images
image_path1 = f"face_detection\c1.jpg"  # Replace with the path to the first image
image_path2 = f"face_detection\c2.jpg"  # Replace with the path to the second image

# Verify if the faces match
result = DeepFace.verify(
    img1_path=image_path1,
    img2_path=image_path2,
    model_name='Facenet512',
    distance_metric='cosine',  # Use cosine similarity 
    enforce_detection=False,  # Skip face detection if faces are already cropped
    threshold=0.2  # Set your custom threshold here lower for stricter results
)

# Print the result
print(result)



