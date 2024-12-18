
from deepface import DeepFace

def embeddings(image_path1,image_path2):
    '''checks if given faces are similar using cosine similarity and Facenet512'''
    # Verify if the faces match
    result = DeepFace.verify(
        img1_path=image_path1,
        img2_path=image_path2,
        model_name='Facenet512',
        distance_metric='cosine',  # Use cosine similarity 
        enforce_detection=False,  # Skip face detection if faces are already cropped
        threshold=0.2  #custom threshold here lower for stricter results
    )
    return result





