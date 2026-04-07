# Libraries
import subprocess
import os
import cv2
import insightface
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0)  

# Training Data directory
known_folder = "training_data"

IMG_SIZE = 224

def gatherinfo():
    known_people = [name for name in os.listdir(known_folder) if os.path.isdir(os.path.join(known_folder, name))]

    attempts = 0
    while attempts < 3:
        x = input("Who do you want to find? ")  
        if x in known_people:
            break
        else:
            print(f"Can't find name '{x}' in training data. Try again.")
            attempts += 1
    else:
        print("Too many invalid attempts. Exiting...")
        exit()

    applescript = '''
    display alert "You'll be prompted to select a folder. Please select the folder" as warning buttons {"OK"}
    '''
    subprocess.run(["osascript", "-e", applescript])

    # Select  folder
    applescript = '''
    set folder_path to choose folder with prompt "Select the folder with images"
    POSIX path of folder_path
    '''
    result = subprocess.run(["osascript", "-e", applescript], capture_output=True, text=True)
    folder_path = result.stdout.strip()
    if not folder_path:
        print("No folder selected. Exiting...")
        exit()
    print("Selected folder:", folder_path)
    return folder_path, x  

# Precompute known embeddings
def load_known_embeddings():
    known_embeddings = {}
    for person_name in os.listdir(known_folder):
        person_path = os.path.join(known_folder, person_name)
        if not os.path.isdir(person_path):
            continue
        for file in os.listdir(person_path):
            if not file.lower().endswith(('.jpg', '.png')):
                continue
            img_path = os.path.join(person_path, file)
            img = np.array(Image.open(img_path).convert("RGB"))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            faces_detected = model.get(img)
            if faces_detected:
                known_embeddings[person_name] = faces_detected[0].embedding
    return known_embeddings

def process_unknown_file(file_path, known_embeddings, threshold):
    unknown_img = np.array(Image.open(file_path).convert("RGB"))
    unknown_img = cv2.resize(unknown_img, (IMG_SIZE, IMG_SIZE))
    unknown_faces = model.get(unknown_img)
    recognized_people = []

    for u_face in unknown_faces:
        u_embedding = u_face.embedding
        match_names = [
            name for name, k_embedding in known_embeddings.items()
            if np.dot(k_embedding, u_embedding) / (np.linalg.norm(k_embedding) * np.linalg.norm(u_embedding)) > threshold
        ]
        recognized_people.append(tuple(match_names if match_names else ["Unknown"]))
    return os.path.basename(file_path), recognized_people

def faces(test_folder, target_name):
    global items, directory_path
    directory_path = test_folder
    known_embeddings = load_known_embeddings()
    threshold = 0.35

    unknown_files = [
        os.path.join(test_folder, f) for f in os.listdir(test_folder)
        if f.lower().endswith(('.jpg', '.png'))
    ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda f: process_unknown_file(f, known_embeddings, threshold), unknown_files))

    items = [
        file_name
        for file_name, recognized_people in results
        if target_name in [name for tup in recognized_people for name in tup]
    ]
    print(f"Images with {target_name}:", items)

def selecting():
    global items, directory_path
    full_paths = [os.path.join(directory_path, f) for f in items if os.path.exists(os.path.join(directory_path, f))]
    if full_paths:
        apple_list = ", ".join([f'POSIX file "{p}"' for p in full_paths])
        applescript = f'''
        tell application "Finder"
            activate
            set target_folder to POSIX file "{directory_path}"
            open target_folder
            set selection to {{{apple_list}}}
        end tell
        '''
        subprocess.run(["osascript", "-e", applescript])
    else:
        print(f"No recognized faces found.")

folder, x = gatherinfo()
faces(folder, x)
selecting()
