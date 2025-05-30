import cv2
import numpy as np
from typing import Tuple
import tempfile
import os
from PIL import Image
import sys
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import streamlit as st

try:
    if getattr(sys, 'frozen', False):
        # En el ejecutable, intentar sys._MEIPASS
        BASE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        #print(f"Executable mode - Initial BASE_DIR: {BASE_DIR} (_MEIPASS: {hasattr(sys, '_MEIPASS')})")
        # Verificar si BASE_DIR contiene los archivos esperados
        expected_dirs = ['navigation', 'models', 'assets', 'img', 'utils']
        if not any(os.path.exists(os.path.join(BASE_DIR, d)) for d in expected_dirs):
            print(f"Warning: Expected directories not found in {BASE_DIR}")
            # Buscar _MEI<random> en el directorio padre
            temp_dir = os.path.dirname(BASE_DIR) if BASE_DIR != os.path.dirname(sys.executable) else BASE_DIR
            for d in os.listdir(temp_dir):
                if d.startswith('_MEI'):
                    candidate = os.path.join(temp_dir, d)
                    if any(os.path.exists(os.path.join(candidate, ed)) for ed in expected_dirs):
                        BASE_DIR = candidate
                        print(f"Adjusted BASE_DIR to _MEI directory: {BASE_DIR}")
                        break
            else:
                print(f"No _MEI directory found in {temp_dir}, using {BASE_DIR}")
    else:
        # En desarrollo, usar el directorio del proyecto
        current_file = os.path.abspath(os.path.realpath(__file__))
        #print(f"Development mode - Current file: {current_file}")
        BASE_DIR = os.path.dirname(os.path.dirname(current_file))  # Subir de utils/ a F1-machine-learning-webapp/
        #print(f"Development mode - BASE_DIR: {BASE_DIR}")
except Exception as e:
    #print(f"Error setting BASE_DIR: {e}")
    # Fallback
    BASE_DIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
    BASE_DIR = os.path.dirname(BASE_DIR)
    #print(f"Fallback BASE_DIR: {BASE_DIR}")

BASE_DIR = os.path.normpath(BASE_DIR)
#print(f"Final BASE_DIR: {BASE_DIR}")



#st.secrets["MONGO_URI"]



# Obtener MONGO_URI de forma segura
def get_mongo_uri():
    try:
        dotenv_path = os.path.join(BASE_DIR, ".env")
        load_dotenv(dotenv_path)
        # Primero intenta desde variables de entorno
        mongo_uri = os.getenv("MONGO_URI")
        if mongo_uri:
            return mongo_uri
    except:
        # Luego intenta desde st.secrets si está disponible
        try:
            return st.secrets.get("MONGO_URI", None)
        except:
            return "a"

#mongo_uri = os.getenv("MONGO_URI")
@st.cache_resource
def get_mongo_client():
    return MongoClient(get_mongo_uri())
    #return MongoClient(st.secrets["MONGO_URI"])
client = get_mongo_client()


def get_metrics_collections():

    db = client["f1_data"]
    metrics_collection = db["usage_metrics"]
    metrics_page = db["visits"]
    return metrics_collection, metrics_page, db

metrics_collection, metrics_page, db = get_metrics_collections()
'''if not metrics_page.find_one({"page": "inicio"}):
    metrics_page.insert_one({"page": "inicio", "visits": 0})
if not metrics_collection.find_one({"action": "descargar_app"}):
    metrics_collection.insert_one({"action": "descargar_app", "count": 0})'''
'''except:
    print("Error loading MongoDB URI from .env file. Please check your configuration.")
    client = None
    metrics_collection = None
    metrics_page = None
    db = None'''


#-------------YOLO ONNX HELPERS-------------------

def preprocess_image_tensor(image_rgb: np.ndarray) -> np.ndarray:
    """Preprocess image to match Ultralytics YOLOv8."""
    
    '''input = np.array(image_rgb)
    input = input.transpose(2, 0, 1)
    input = input.reshape(1,3,224,224).astype("float32")
    input = input/255.0'''

    input_data = image_rgb.transpose(2, 0, 1).reshape(1, 3, 224, 224)

    # Convert to float32 and normalize to [0, 1]
    input_data = input_data.astype(np.float32) / 255.0
    
    return input_data

def postprocess_outputs(outputs: list, height: int, width: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process ONNX model outputs for a single-class model."""
    res_size = 56
    output0 = outputs[0]
    output1 = outputs[1]

    output0 = output0[0].transpose()
    output1 = output1[0]

    boxes = output0[:,0:5]
    masks = output0[:,5:]

    output1 = output1.reshape(32,res_size*res_size)

    masks = masks @ output1

    boxes = np.hstack([boxes,masks])

    yolo_classes = [
        "helmet"
    ]

    # parse and filter all boxes
    objects = []
    for row in boxes:
        xc,yc,w,h = row[:4]
        x1 = (xc-w/2)/224*width
        y1 = (yc-h/2)/224*height
        x2 = (xc+w/2)/224*width
        y2 = (yc+h/2)/224*height
        prob = row[4:5].max()
        if prob < 0.2:
            continue
        class_id = row[4:5].argmax()
        label = yolo_classes[class_id]

        mask = get_mask(row[5:25684], (x1,y1,x2,y2), width, height)
        try:
            polygon = get_polygon(mask)
        except:
            continue
        objects.append([x1,y1,x2,y2,label,prob,mask,polygon])



    # apply non-maximum suppression
    objects.sort(key=lambda x: x[5], reverse=True)
    result = []
    while len(objects)>0:
        result.append(objects[0])
        objects = [object for object in objects if iou(object,objects[0])<0.7]



    return True,result

def intersection(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1) 

def union(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)

def iou(box1,box2):
    return intersection(box1,box2)/union(box1,box2)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# parse segmentation mask
def get_mask(row, box, img_width, img_height):
    # convert mask to image (matrix of pixels)
    res_size = 56
    mask = row.reshape(res_size,res_size)
    mask = sigmoid(mask)
    mask = (mask > 0.2).astype("uint8")*255
    # crop the object defined by "box" from mask
    x1,y1,x2,y2 = box
    mask_x1 = round(x1/img_width*res_size)
    mask_y1 = round(y1/img_height*res_size)
    mask_x2 = round(x2/img_width*res_size)
    mask_y2 = round(y2/img_height*res_size)
    mask = mask[mask_y1:mask_y2,mask_x1:mask_x2]
    # resize the cropped mask to the size of object
    img_mask = Image.fromarray(mask,"L")
    img_mask = img_mask.resize((round(x2-x1),round(y2-y1)))
    mask = np.array(img_mask)
    return mask



# calculate bounding polygon from mask
def get_polygon(mask):
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polygon = [[contour[0][0],contour[0][1]] for contour in contours[0][0]]
    return polygon










#------------------VIDEO CONVERSION------------------

def convert_video_to_10fps(video_file):
    """
    Convert an uploaded video file to 10 FPS and return metadata
    
    Args:
        video_file: Streamlit uploaded file object
        
    Returns:
        Dictionary with video metadata and path to converted file
    """
    try:
        # Create temporary file for the original upload
        orig_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        orig_tfile.write(video_file.read())
        orig_tfile.close()
        
        # Open the original video to get properties
        orig_cap = cv2.VideoCapture(orig_tfile.name)
        
        if not orig_cap.isOpened():
            return {"success": False, "error": "Could not open video file"}
            
        orig_fps = orig_cap.get(cv2.CAP_PROP_FPS)
        width = int(orig_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(orig_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_total_frames = int(orig_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate duration
        duration_seconds = orig_total_frames / orig_fps
        expected_frames = int(duration_seconds * 10)  # 10 fps
        
        # Create output temp file
        converted_path = tempfile.mktemp(suffix='.mp4')
        
        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(converted_path, fourcc, 10, (width, height))
        
        # Calculate frame sampling
        if orig_fps <= 10:
            # If original is slower than target, duplicate frames
            step = 1
            duplication = int(10 / orig_fps)
        else:
            # If original is faster, skip frames
            step = orig_fps / 10
            duplication = 1
            
        # Convert the video
        frame_count = 0
        output_count = 0
        
        while orig_cap.isOpened():
            ret, frame = orig_cap.read()
            if not ret:
                break
                
            # Determine if we should include this frame
            if frame_count % step < 1:  # Using modulo < 1 for floating point step values
                # Write frame (possibly multiple times)
                for _ in range(duplication):
                    out.write(frame)
                    output_count += 1
                    
            frame_count += 1
            
        # Release resources
        orig_cap.release()
        out.release()
        os.unlink(orig_tfile.name)  # Delete original temp file
        
        # Instead of returning a dictionary, read the file back into memory
        with open(converted_path, "rb") as f:
            video_data = f.read()
        
        # Clean up the temporary file
        os.unlink(converted_path)
        
        # Return a file-like object
        from io import BytesIO
        video_io = BytesIO(video_data)
        video_io.name = "converted_10fps.mp4"
        return video_io
        
    except Exception as e:
        print(f"Error converting video: {e}")
        return None

def recortar_imagen(image,starty_dic, axes_dic):
    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    start_y = int((starty_dic-.02) * height)
    cv2.rectangle(mask, (0, start_y), (width, height), 255, -1)
    center = (width // 2, start_y)
    axes = (width // 2, int(axes_dic * height))
    cv2.ellipse(mask, center, axes, 0, 180, 360, 255, -1)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def recortar_imagen_again(image,starty_dic, axes_dic):
    
    try:
        height, width,_ = image.shape
    except :
        height, width = image.shape

    mask = np.zeros((height, width), dtype=np.uint8)

    start_y = int(starty_dic * height)
    cv2.rectangle(mask, (0, start_y), (width, height), 255, -1)
    center = (width // 2, start_y)
    axes = (width // 2, int(axes_dic * height))
    cv2.ellipse(mask, center, axes, 0, 180, 360, 255, -1)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def calculate_black_pixels_percentage(image):
    """
    Calcula el porcentaje de píxeles totalmente negros en la imagen.
    
    Args:
        image: Imagen cargada con cv2 (BGR o escala de grises).
        is_grayscale: True si la imagen ya está en escala de gruises, False si es a color.
    
    Returns:
        float: Porcentaje de píxeles negros.
    """
    # Obtener dimensiones
    '''image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)'''
    if image is None:
        print(f"Error loading image")
        return 0
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = image.copy()
    h, w = image.shape[:2]
    total_pixels = h * w
    
    black_pixels = np.sum(image < 10)

    # Calcular porcentaje
    percentage = (black_pixels / total_pixels) * 100

    
    percentage = (100.00 - float(percentage)) * .06

    
    return percentage

def create_rectangular_roi(height, width, x1=0, y1=0, x2=None, y2=None):
    if x2 is None:
        x2 = width
    if y2 is None:
        y2 = height
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask

def preprocess_image(image, mask=None):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    denoised = cv2.bilateralFilter(gray, d=3, sigmaColor=20, sigmaSpace=10)
    sharpened = cv2.addWeighted(denoised, 3.0, denoised, -2.0, 0)
    normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
    
    if mask is not None:
        return cv2.bitwise_and(normalized, normalized, mask=mask)
    return normalized

def calculate_robust_rms_contrast(image, mask=None, bright_threshold=240):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is not None:
        masked_image = image[mask > 0]
    else:
        masked_image = image.ravel()
    
    if len(masked_image) == 0:
        mean = np.mean(image)
        std_dev = np.sqrt(np.mean((image - mean) ** 2))
    else:
        mask_bright = masked_image < bright_threshold
        masked_image = masked_image[mask_bright]
        if len(masked_image) == 0:
            mean = np.mean(image)
            std_dev = np.sqrt(np.mean((image - mean) ** 2))
        else:
            mean = np.mean(masked_image)
            std_dev = np.sqrt(np.mean((masked_image - mean) ** 2))
    return std_dev / 255.0

def adaptive_clahe_iterative(image, roi_mask, initial_clip_limit=1.0, max_clip_limit=10.0, iterations=20, target_rms_min=0.199, target_rms_max=0.5, bright_threshold=230):
    if len(image.shape) == 3:
        original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = image.copy()
    
    #preprocessed_image = preprocess_image(original_gray)
    
    best_image = original_gray.copy()
    best_rms = calculate_robust_rms_contrast(original_gray, roi_mask, bright_threshold)
    clip_limit = initial_clip_limit
    
    for i in range(iterations):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        current_image = clahe.apply(original_gray)
        
        rms_contrast = calculate_robust_rms_contrast(current_image, roi_mask, bright_threshold)
        
        if target_rms_min <= rms_contrast <= target_rms_max:
            return current_image   
        if rms_contrast > best_rms:
            best_rms = rms_contrast
            best_image = current_image.copy()
        if rms_contrast > target_rms_max:
            clip_limit = min(clip_limit, 1.0)
        else:
            clip_limit = min(initial_clip_limit + (i * 0.5), max_clip_limit)
    
    return best_image

def adaptive_edge_detection(imagen, min_edge_percentage=5.5, max_edge_percentage=6.5, target_percentage=6.0, max_attempts=5,mode="Default"):
    """
    Detecta bordes con ajuste progresivo de parámetros hasta lograr un porcentaje óptimo
    de píxeles de borde en la imagen - optimizado con operaciones vectorizadas.
    """
    # Read image
    original = imagen
    if original is None:
        print(f"Error loading image")
        return None, None, None, None
    
    # Convert to grayscale
    gray = original
    
    # Calculate total pixels for percentage calculation
    total_pixels = gray.shape[0] * gray.shape[1]
    min_edge_pixels = int((min_edge_percentage / 100) * total_pixels)
    max_edge_pixels = int((max_edge_percentage / 100) * total_pixels)
    target_edge_pixels = int((target_percentage / 100) * total_pixels)
    
    # Initial parameters - ajustados para conseguir un rango alrededor del 6% de bordes
    clip_limits = [1]
    grid_sizes = [(2, 2)]
    # Empezamos con umbrales más altos para restringir la cantidad de bordes
    canny_thresholds = [(55, 170), (45, 160), (35, 150), (25, 140), (20, 130),(20, 130),(20, 130)]
    
    best_edges = None
    best_enhanced = None
    best_config = None
    best_edge_score = float('inf')  # Inicializamos con un valor alto
    edge_percentage = 0

    
    # Try progressively more aggressive parameters
    for attempt in range(max_attempts):
        # Get parameters for this attempt
        clip_limit = clip_limits[attempt]
        grid_size = grid_sizes[attempt]
        low_threshold, high_threshold = canny_thresholds[attempt]
        
        if edge_percentage <= max_edge_percentage:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        elif edge_count > max_edge_percentage:
            # Si hay demasiados bordes, aplicamos un CLAHE más fuerte
            clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=grid_size)
        
        enhanced = clahe.apply(gray)

        
        #print("denoised shape:", denoised.shape, "dtype:", denoised.dtype)
        # Apply noise reduction for higher attempts
        '''if attempt >= 2:
            enhanced = cv2.bilateralFilter(enhanced, 5, 100, 100)'''
        

        
        if mode == "Default":
            denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=200, sigmaSpace=200)
            median_intensity = np.median(denoised)
            low_threshold = max(20, (1.0 - .3) * median_intensity)
            high_threshold = max(80, (1.0 + .8) * median_intensity)
        elif mode == "Low ilumination":
            denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=200, sigmaSpace=200)
            median_intensity = np.median(denoised)
            low_threshold = max(20, (1.0 - .3) * median_intensity)
            high_threshold = max(80, (1.0 + .8) * median_intensity)
        # Edge detection
        
        edges = cv2.Canny(denoised, low_threshold, high_threshold)
        std_intensity = np.std(edges)
        
        # Reducir ruido con operaciones morfológicas - vectorizado
        kernel = np.ones((1, 1), np.uint8)
        edges = cv2.morphologyEx(
            edges, 
            cv2.MORPH_OPEN, 
            kernel, 
            iterations=0 if std_intensity < 60 else 1  # Más iteraciones si hay más ruido
        )
        
        
        # Count edge pixels - vectorizado usando np.count_nonzero
        edge_count = np.count_nonzero(edges)
        edge_percentage = (edge_count / total_pixels) * 100
        
        # Calcular distancia al objetivo - vectorizado
        edge_score = abs(edge_count - target_edge_pixels)
        
        # Record the best attempt (closest to target percentage)
        if edge_score < best_edge_score:
            best_edge_score = edge_score
            best_edges = edges.copy()  # Hacer copia para evitar sobrescrituras
            best_enhanced = enhanced.copy()
            best_config = {
                'attempt': attempt + 1,
                'clip_limit': clip_limit,
                'grid_size': grid_size,
                'canny_thresholds': (low_threshold, high_threshold),
                'edge_pixels': edge_count,
                'edge_percentage': edge_percentage
            }
        
        # Salida temprana si estamos cerca del objetivo
        if abs(edge_percentage - target_percentage) < 0.1:  # Within 0.2% of target
            break
    
    #print(f"Mejor intento: {best_config['attempt']}, porcentaje de bordes: {edge_percentage:.2f}%")
    return best_enhanced, best_edges, original, best_config
