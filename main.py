import subprocess
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import insightface
import onnxruntime
onnxruntime.set_default_logger_severity(3)

from insightface.model_zoo import model_zoo
from insightface.app.common import Face

# Detection Model - CPU
det_path = os.path.expanduser('~/.insightface/models/buffalo_l/det_10g.onnx')
det_model = model_zoo.get_model(det_path, providers=['CPUExecutionProvider'])
det_model.prepare(ctx_id=0, input_size=(480, 480))

# Recognition Model - CoreML/GPU
rec_path = os.path.expanduser('~/.insightface/models/buffalo_l/w600k_r50.onnx')
rec_model = model_zoo.get_model(rec_path, providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
rec_model.prepare(ctx_id=0)

known_folder = "training_data"

def gatherinfo():
    if not os.path.exists(known_folder):
        exit(f"Directory not found: {known_folder}")

    known_people = [name for name in os.listdir(known_folder) if os.path.isdir(os.path.join(known_folder, name))]
    search_names = []

    while True:
        name = input("Who do you want to find? ")  
        if name in known_people:
            search_names.append(name)
        else:
            print(f"Can't find name '{name}' in training data.")
            continue

        more = input("Do you want to search for another person? (y/n): ").strip().lower()
        if more != "y":
            break

    applescript = '''
    display alert "You'll be prompted to select a folder. Please select the folder" as warning buttons {"OK"}
    set folder_path to choose folder with prompt "Select the folder with images"
    POSIX path of folder_path
    '''
    result = subprocess.run(["osascript", "-e", applescript], capture_output=True, text=True)
    folder_path = result.stdout.strip()
    
    if not folder_path:
        exit("No folder selected. Exiting...")
        
    print("Selected folder:", folder_path)
    return folder_path, search_names  

def get_faces(img):
    """Custom function to route through our split CPU/GPU models"""
    bboxes, kpss = det_model.detect(img, max_num=0, metric='default')
    faces = []
    if bboxes is None:
        return faces
        
    for i in range(bboxes.shape[0]):
        face = Face(bbox=bboxes[i, 0:4], kps=kpss[i])
        rec_model.get(img, face)
        faces.append(face)
    return faces

def load_known_embeddings():
    known_embeddings = {}
    for person_name in os.listdir(known_folder):
        person_path = os.path.join(known_folder, person_name)
        if not os.path.isdir(person_path): continue
        
        for file in os.listdir(person_path):
            if file.lower().endswith(('.jpg', '.png')):
                img = cv2.imread(os.path.join(person_path, file))
                if img is None: continue
                
                faces = get_faces(img)
                if faces:
                    emb = faces[0].embedding
                    known_embeddings[person_name] = emb / np.linalg.norm(emb)
                break 
    return known_embeddings

def faces(test_folder, target_names_list):
    known_dict = load_known_embeddings()
    target_vectors = [known_dict[name] for name in target_names_list if name in known_dict]
    
    if not target_vectors:
        print("No valid face data found for your targets.")
        return []

    threshold = 0.45 
    found_paths = []
    
    test_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) 
                  if f.lower().endswith(('.jpg', '.png'))]

    print(f"Scanning {len(test_files)} images using CPU Detection + GPU Recognition...")
    
    def check_image(f_path):
        img = cv2.imread(f_path)
        if img is None: return None
        
        height, width = img.shape[:2]
        if width > 1920:
            img = cv2.resize(img, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
            
        detected_faces = get_faces(img)
        for face in detected_faces:
            u_emb = face.embedding / np.linalg.norm(face.embedding)
            
            for t_vec in target_vectors:
                if np.dot(t_vec, u_emb) > threshold:
                    return f_path 
        return None

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = executor.map(check_image, test_files)
        for res in results:
            if res is not None:
                found_paths.append(res)
                
    return found_paths 

def selecting(items, directory_path):
    if not items:
        print("No recognized faces found.")
        return

    full_paths = [os.path.join(directory_path, f) for f in items if os.path.exists(f)]
    
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
        print(f"Successfully selected {len(full_paths)} matches in Finder.")


folder, names_list = gatherinfo()
matched_files = faces(folder, names_list)
selecting(matched_files, folder)
