
import is_biometric as ib
import measure_distance as md
import embeddings as em

'''pip freeze > requirements.txt '''

def check_pic(image,height, width,landmark_dict):

    bigger_than_90 = ib.calculate_inter_eye_distance(landmark_dict,width,height,90)
    left_eye = ib.check_eye_open(landmark_dict,width,height,"left")
    right_eye = ib.check_eye_open(landmark_dict,width,height,"right")

    tilt = ib.check_tilt(landmark_dict, image_height=height, image_width=width)
    
    sym = ib.check_face_symmetry(landmark_dict,width,height,185)

    features_present = ib.check_features(landmark_dict)
    
    back = ib.check_background(image)
    focus = ib.is_in_focus(image,12) 
    noise = ib.check_noise(image)
    face_area = ib.check_face_area(image,50)
    if not bigger_than_90:
        return "img res too low"
    elif not left_eye:
        return "left eye closed"
    elif not right_eye:
        return "right eye closed"
    elif not tilt:
        return "face too tilted"
    elif not sym:
        return "unusual facial symmetry"
    elif not features_present:
        return "required facial features not found"
    elif not back:
        return "image background not uniform"
    elif not focus:
        return "image out of focus"
    elif not noise:
        return "image is too noisy"
    elif not face_area:
        return "face area is too little"
    else :return True

def get_distances(landmark_dict):
    distances=[]
    distances.append(md.interpupilary(landmark_dict=landmark_dict))   
    distances.append(md.eye_to_eye(landmark_dict=landmark_dict))
    distances.append(md.eye_to_mouth(landmark_dict=landmark_dict))
    distances.append(md.eye_to_chin(landmark_dict=landmark_dict))
    distances.append(md.nose_to_mouth(landmark_dict=landmark_dict))
    distances.append(md.nose_to_chin(landmark_dict=landmark_dict))
    distances.append(md.mouth_width(landmark_dict=landmark_dict))
    distances.append(md.mouth_to_chin(landmark_dict=landmark_dict))
    distances.append(md.face_height(landmark_dict=landmark_dict))
    distances.append(md.ear_to_ear(landmark_dict=landmark_dict))
    distances.append(md.eyebrow_to_eye(landmark_dict=landmark_dict))
    distances.append(md.eyebrow_len(landmark_dict=landmark_dict))
    distances.append(md.forehead_height(landmark_dict=landmark_dict))
    distances.append(md.jawline_width(landmark_dict=landmark_dict))

    return distances

def compare_distances(distances1,distances2,threhold=0.04):
    '''checks if distances are bigger than threshold if bigger the faces aren't similar'''
    result = []
    for a, b in zip(distances1, distances2):
        if isinstance(a, tuple) and isinstance(b, tuple):
        # Subtract tuples element-wise and take absolute values
            result.append(tuple(abs(x - y) for x, y in zip(a, b)))
        else:
            # Subtract single values and take absolute value
            result.append(abs(a - b))


    exceeds_threshold = any(
    max(value) > threhold if isinstance(value, tuple) else value > threhold
    for value in result)

    return not exceeds_threshold


def get_facenet_result(image_path1,image_path2):
    '''checks facenet for facial similarity returns true if similar'''
    return em.embeddings(image_path1,image_path2)

def main():
    image = ib.load_image(f"face_detection\\tt2.JPG")
    image2 = ib.load_image(f"face_detection\\tt3.JPG")


    
    height, width,channel = image.shape
    img_dict = ib.get_facial_features(image)

    
    
    height2, width2,channel2 = image2.shape
    img_dict2=ib.get_facial_features(image2)

    p1c = (check_pic(image,height,width,img_dict))
    p2c = (check_pic(image2,height,width,img_dict))

    if p1c!=True:
        print(f"pick 1 not suitable: {p1c}" )
    
    elif p2c!=True:
        print(f"pick 2 not suitable: {p2c}" )


    else:
        r1=get_distances(img_dict)
        r2=get_distances(img_dict2)
        res = compare_distances(r1,r2)
        
        facenet=get_facenet_result(image,image2)
        if facenet and res:
            print("the faces match")
            
        else:
            print("no match found")
            print(facenet)
            print(res) 
    
    


if __name__ == "__main__":
    main()