import face_recognition
import matplotlib.pyplot as plt
import glob
import os

def get_encoding(img_path):
    img = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(img)
    #[0] becase I know there is only one face in the image
    return encoding[0]

known_faces = []
known_names = []
known_faces_paths = []

registered_faces_path = 'img/registered/'

for name in os.listdir(registered_faces_path):
    img_mask = '%s%s/*.jpg' % (registered_faces_path, name)
    img_paths = glob.glob(img_mask)
    known_faces_paths += img_paths
    known_names += [name for x in img_paths ]
    
   
known_faces = [get_encoding(img_path) for img_path in known_faces_paths ]

test_imgs = glob.glob('img/test/*.jpg')

for img_path in test_imgs:
    img = plt.imread(img_path)
    
    plt.figure()
    plt.imshow(img)
    
    encodings = face_recognition.face_encodings(img) 
    
    found_faces = []
    
    for face_code in encodings:
        results = face_recognition.compare_faces(known_faces, face_code, tolerance = 0.6)
        if any(results):
            found_faces.append(known_names[results.index(True)])
        else: 
            found_faces.append("No one")
            
            
    plt.title(found_faces)