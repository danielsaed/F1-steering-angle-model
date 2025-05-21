import cv2
import numpy as np
from typing import List, Tuple
import tempfile
import os
import time
import functools
from collections import defaultdict
import onnxruntime as ort
from PIL import Image


class Profiler:
    """Clase para trackear el tiempo de ejecución de las funciones"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Profiler, cls).__new__(cls)
            cls._instance.function_times = defaultdict(list)
            cls._instance.call_counts = defaultdict(int)
        return cls._instance
    
    def track_time(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed = end_time - start_time
            
            self.function_times[func.__name__].append(elapsed)
            self.call_counts[func.__name__] += 1
            
            return result
        return wrapper
    
    def print_stats(self):
        print("\n===== FUNCIÓN TIMING STATS =====")
        print(f"{'FUNCIÓN':<30} {'LLAMADAS':<10} {'TOTAL (s)':<15} {'PROMEDIO (s)':<15} {'% TIEMPO':<10}")
        
        total_time = sum(sum(times) for times in self.function_times.values())
        
        # Ordenar por tiempo total (descendente)
        sorted_funcs = sorted(
            self.function_times.items(),
            key=lambda x: sum(x[1]),
            reverse=True
        )
        
        for func_name, times in sorted_funcs:
            total = sum(times)
            avg = total / len(times) if times else 0
            calls = self.call_counts[func_name]
            percent = (total / total_time * 100) if total_time > 0 else 0
            
            print(f"{func_name:<30} {calls:<10} {total:<15.4f} {avg:<15.4f} {percent:<10.2f}%")
        
        print(f"\nTiempo total de procesamiento: {total_time:.4f} segundos")
        print("================================")
    
    def get_stats_dict(self):
        """Devuelve las estadísticas como un diccionario para mostrar en Streamlit"""
        stats = []
        total_time = sum(sum(times) for times in self.function_times.values())
        
        for func_name, times in self.function_times.items():
            total = sum(times)
            avg = total / len(times) if times else 0
            calls = self.call_counts[func_name]
            percent = (total / total_time * 100) if total_time > 0 else 0
            
            stats.append({
                'función': func_name,
                'llamadas': calls,
                'tiempo_total': total,
                'tiempo_promedio': avg,
                'porcentaje': percent
            })
        
        # Ordenar por porcentaje de tiempo
        stats.sort(key=lambda x: x['porcentaje'], reverse=True)
        return stats, total_time
    
    def reset(self):
        """Reiniciar las estadísticas"""
        self.function_times.clear()
        self.call_counts.clear()

profiler = Profiler()

@profiler.track_time
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
        polygon = get_polygon(mask)
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

# Funciones previas sin cambios (recortar_imagen, create_rectangular_roi, preprocess_image, calculate_robust_rms_contrast, adaptive_clahe_iterative)
@profiler.track_time
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

@profiler.track_time
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

@profiler.track_time
def adaptive_edge_detection(imagen, min_edge_percentage=5.5, max_edge_percentage=6.5, target_percentage=6.0, max_attempts=5):
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
    clip_limits = [1, 1, 1, 1, 1, 1, 1]
    grid_sizes = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
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

        denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=200, sigmaSpace=200)
        #print("denoised shape:", denoised.shape, "dtype:", denoised.dtype)
        # Apply noise reduction for higher attempts
        '''if attempt >= 2:
            enhanced = cv2.bilateralFilter(enhanced, 5, 100, 100)'''
        

        median_intensity = np.median(denoised)
        low_threshold = max(30, (1.0 -.4) * median_intensity)
        high_threshold = max(90, (1.0 + 0.5) * median_intensity)
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
        
        # Vectorización del ajuste fino:
        # En lugar de usar bucles anidados para ajustar los umbrales, podemos
        # calcular todos los posibles umbrales ajustados de una vez y aplicar
        # el primero que dé un resultado satisfactorio.
        
        '''if edge_count > max_edge_pixels:  # Si hay demasiados bordes (>6.5%)
            # Vectorizar el cálculo de nuevos umbrales
            threshold_adjustments = np.arange(1, 13)  # 1 a 12
            new_lows = low_threshold + 5 * threshold_adjustments
            new_highs = high_threshold + 5 * threshold_adjustments
            
            # Buscar el primer ajuste que funcione
            adjusted_edges = None
            for i in range(len(threshold_adjustments)):
                # Volver a aplicar Canny con umbrales ajustados
                temp_edges = cv2.Canny(enhanced, new_lows[i], new_highs[i])
                
                if morph_iterations > 0:
                    temp_edges = cv2.morphologyEx(temp_edges, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
                
                temp_count = np.count_nonzero(temp_edges)
                
                if temp_count <= max_edge_pixels:
                    edges = temp_edges
                    edge_count = temp_count
                    edge_percentage = (edge_count / total_pixels) * 100
                    break
                    
        elif edge_count < min_edge_pixels:  # Si hay muy pocos bordes (<5.5%)
            # Vectorizar el cálculo de nuevos umbrales (disminuyendo)
            threshold_adjustments = np.arange(1, 13)  # 1 a 12
            new_lows = np.maximum(10, low_threshold - 5 * threshold_adjustments)
            new_highs = np.maximum(30, high_threshold - 5 * threshold_adjustments)
            
            # Buscar el primer ajuste que funcione
            for i in range(len(threshold_adjustments)):
                # Volver a aplicar Canny con umbrales ajustados
                temp_edges = cv2.Canny(enhanced, new_lows[i], new_highs[i])
                
                if morph_iterations > 0:
                    temp_edges = cv2.morphologyEx(temp_edges, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
                
                temp_count = np.count_nonzero(temp_edges)
                
                if min_edge_pixels <= temp_count <= max_edge_pixels:
                    edges = temp_edges
                    edge_count = temp_count
                    edge_percentage = (edge_count / total_pixels) * 100
                    break
        '''
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
    
    print(f"Mejor intento: {best_config['attempt']}, porcentaje de bordes: {edge_percentage:.2f}%")
    return best_enhanced, best_edges, original, best_config

class VideoProcessor:
    def __init__(self):
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.target_fps = 10
        self.driver_crop_type = "albon"
        self.load_crop_variables(self.driver_crop_type)
        #self.yolo_model = YOLO("models/best.pt")
        self.model = ort.InferenceSession("models/best-224.onnx")
        self.input_shape = (224, 224)  # Match imgsz=224 from your original code
        self.conf_thres = 0.5  # Confidence threshold
        self.iou_thres = 0.5   # IoU threshold for NMS
    
    @profiler.track_time
    def load_crop_variables(self,driver_crop_type):
        """
        Cargar variables de recorte según el tipo de conductor
        """
        driver_config = {
    "albon": {
        "starty": 0.55,
        "axes": 0.39,
        "y_start": 0.53,
        "x_center": 0.59
    },
    "alonso": {
        "starty": 0.5,
        "axes": 0.29,
        "y_start": 0.53,
        "x_center": 0.56
    },
    "bottas": {
        "starty": 0.67,
        "axes": 0.43,
        "y_start": 0.53,
        "x_center": 0.574
    },
    "colapinto": {
        "starty": 0.52,
        "axes": 0.33,
        "y_start": 0.53,
        "x_center": 0.594
    },
    "hamilton-arabia": {
        "starty": 0.908,
        "axes": 0.4,
        "y_start": 0.53,
        "x_center": 0.554
    },
     "Hamilton 2025": {
        "starty": 0.59,
        "axes": 0.4,
        "y_start": 0.53,
        "x_center": 0.573
    },

    "hamilton-texas": {
        "starty": 0.7,
        "axes": 0.38,
        "y_start": 0.53,
        "x_center": 0.6
    },
   "leclerc-china": {
        "starty": 0.6,
        "axes": 0.36,
        "y_start": 0.53,
        "x_center": 0.58
    },
    
    "Leclerc 2025": {
        "starty": 0.65,
        "axes": 0.45,
        "y_start": 0.53,
        "x_center": 0.575
    },
    "magnussen": {
        "starty": 0.6,
        "axes": 0.34,
        "y_start": 0.53,
        "x_center": 0.58
    },
    "norris-arabia": {
        "starty": 0.7,
        "axes": 0.3,
        "y_start": 0.53,
        "x_center": 0.58
    },
    "norris-texas": {
        "starty": 0.7,
        "axes": 0.3,
        "y_start": 0.53,
        "x_center": 0.58
    },
    "Norris 2025": {
        "starty": 0.79,
        "axes": 0.6,
        "y_start": 0.53,
        "x_center": 0.571,
        "helmet_height_ratio": 0.5
    },
    "ocon": {
        "starty": 0.75,
        "axes": 0.35,
        "y_start": 0.53,
        "x_center": 0.555
    },
    "piastri-azerbaiya": {
        "starty": 0.65,
        "axes": 0.34,
        "y_start": 0.53,
        "x_center": 0.549
    },
    "piastri-singapure": {
        "starty": 0.65,
        "axes": 0.34,
        "y_start": 0.53,
        "x_center": 0.549
    },
    'Piastri 2025': {
        "starty": 0.93,
        "axes": 0.59,
        "y_start": 0.53,
        "x_center": 0.573,
        "helmet_height_ratio": 0.3
    },
    "russel-singapure": {
        "starty": 0.63,
        "axes": 0.44,
        "y_start": 0.53,
        "x_center": 0.56
    },
    "Russell 2025": {
        "starty": 0.95,
        "axes": 0.65,
        "y_start": 0.53,
        "x_center": 0.574,
        "helmet_height_ratio": 0.35
    },
    "sainz": {
        "starty": 0.57,
        "axes": 0.32,
        "y_start": 0.53,
        "x_center": 0.59
    },

    
    "Tsunoda 2025":{
        "starty": 0.92,
        "axes": 0.55,
        "y_start": 0.53,
        "x_center": 0.58,
        "helmet_height_ratio": 0.25
    },
    "verstappen_china": {
        "starty": 0.7,
        "axes": 0.42,
        "y_start": 0.53,
        "x_center": 0.57
    },
    "Verstappen 2025": {
        "starty": 0.7,
        "axes": 0.42,
        "y_start": 0.53,
        "x_center": 0.57,
        "helmet_height_ratio": 0.4
    },
    "vertappen": {
        "starty": 0.7,
        "axes": 0.42,
        "y_start": 0.53,
        "x_center": 0.57
    },
    "verstappen-arabia": {
        "starty": 0.95,
        "axes": 0.4,
        "y_start": 0.53,
        "x_center": 0.565
    },
    "yuki": {
        "starty": 0.64,
        "axes": 0.37,
        "y_start": 0.53,
        "x_center": 0.585
    },
    "Antonelli 2025":
    {
        "starty": 0.97,
        "axes": 0.65,
        "y_start": 0.53,
        "x_center": 0.595,
        "helmet_height_ratio": 0.5
    }}
        
        print(f"Driver crop type: {self.driver_crop_type}")
        self.driver_crop_type = driver_crop_type
        self.starty = driver_config[self.driver_crop_type]["starty"]
        self.axes = driver_config[self.driver_crop_type]["axes"]

        self.y_start = driver_config[self.driver_crop_type]["y_start"]
        self.x_center = driver_config[self.driver_crop_type]["x_center"]
        self.helmet_height_ratio = driver_config[self.driver_crop_type]["helmet_height_ratio"] if "helmet_height_ratio" in driver_config[self.driver_crop_type] else 0.5

    @profiler.track_time
    def load_video(self, video_file) -> bool:
        """Load video file and get basic information"""
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        # Guardar ruta para posibles reinicios
        self.video_path = tfile.name
        
        self.cap = cv2.VideoCapture(tfile.name)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        return True
            
    @profiler.track_time
    def get_frame(self, frame_number: int) -> np.ndarray:
        """
        Obtiene un frame específico del video con optimizaciones de rendimiento
        
        Args:
            frame_number: Número del frame a obtener
            
        Returns:
            Frame como array NumPy (formato RGB) o None si no está disponible
        """
        if self.cap is None:
            return None
        
        # 1. Inicializar atributos de seguimiento si no existen
        if not hasattr(self, 'frame_cache'):
            # Usamos un diccionario limitado para caché de frames frecuentes
            self.frame_cache = {}
            self.frame_cache_size = 100  # Ajustar según memoria disponible
            self.last_position = -1  # Para seguimiento de posición
        
        # 2. Consultar caché primero (mejora extrema para frames accedidos repetidamente)
        if frame_number in self.frame_cache:
            return self.frame_cache[frame_number]
        
        # 3. Optimización para acceso secuencial (evita seeks innecesarios)
        if hasattr(self, 'last_position') and frame_number == self.last_position + 1:
            # El frame solicitado es el siguiente al último leído
            ret, frame = self.cap.read()
            if ret:
                self.last_position = frame_number
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #rgb_frame = frame
                
                # Añadir al caché
                self.frame_cache[frame_number] = rgb_frame
                
                # Mantener tamaño del caché
                if len(self.frame_cache) > self.frame_cache_size:
                    # Eliminar el frame más antiguo (menor número)
                    oldest = min(self.frame_cache.keys())
                    del self.frame_cache[oldest]
                    
                return rgb_frame
            # Si falla la lectura, continuar con método directo
        
        # 4. Acceso directo con mecanismo de reintento
        for attempt in range(3):  # Intentar hasta 3 veces si falla
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            
            if ret:
                # Actualizar last_position para futuras optimizaciones secuenciales
                self.last_position = frame_number
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Añadir al caché
                self.frame_cache[frame_number] = rgb_frame
                
                # Mantener tamaño del caché
                if len(self.frame_cache) > self.frame_cache_size:
                    # Eliminar el frame más antiguo (menor número)
                    oldest = min(self.frame_cache.keys())
                    del self.frame_cache[oldest]
                    
                return rgb_frame
                
            if attempt < 2:  # No reintentar en el último intento
                # Restaurar el objeto cap en caso de error
                # Esto ayuda con formatos de video problemáticos
                if hasattr(self, 'video_path') and self.video_path:
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.video_path)
        
        # Si llegamos aquí, todos los intentos fallaron
        return None
    
    def get_frame_example(self, frame_number: int) -> np.ndarray:
        """
        Obtiene un frame específico del video con optimizaciones de rendimiento
        
        Args:
            frame_number: Número del frame a obtener
            
        Returns:
            Frame como array NumPy (formato RGB) o None si no está disponible
        """
        if self.cap is None:
            return None
        
        # 1. Inicializar atributos de seguimiento si no existen
        if not hasattr(self, 'frame_cache'):
            # Usamos un diccionario limitado para caché de frames frecuentes
            self.frame_cache = {}
            self.frame_cache_size = 30  # Ajustar según memoria disponible
            self.last_position = -1  # Para seguimiento de posición
        
        # 2. Consultar caché primero (mejora extrema para frames accedidos repetidamente)
        if frame_number in self.frame_cache:
            return self.frame_cache[frame_number]
        
        # 3. Optimización para acceso secuencial (evita seeks innecesarios)
        if hasattr(self, 'last_position') and frame_number == self.last_position + 1:
            # El frame solicitado es el siguiente al último leído
            ret, frame = self.cap.read()
            if ret:
                self.last_position = frame_number
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Añadir al caché
                self.frame_cache[frame_number] = rgb_frame
                
                # Mantener tamaño del caché
                if len(self.frame_cache) > self.frame_cache_size:
                    # Eliminar el frame más antiguo (menor número)
                    oldest = min(self.frame_cache.keys())
                    del self.frame_cache[oldest]
                
                rgb_frame = self.crop_frame(rgb_frame)
                return rgb_frame
            # Si falla la lectura, continuar con método directo
        
        # 4. Acceso directo con mecanismo de reintento
        for attempt in range(3):  # Intentar hasta 3 veces si falla
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            
            if ret:
                # Actualizar last_position para futuras optimizaciones secuenciales
                self.last_position = frame_number
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Añadir al caché
                self.frame_cache[frame_number] = rgb_frame
                
                # Mantener tamaño del caché
                if len(self.frame_cache) > self.frame_cache_size:
                    # Eliminar el frame más antiguo (menor número)
                    oldest = min(self.frame_cache.keys())
                    del self.frame_cache[oldest]
                    
                return rgb_frame
                
            if attempt < 2:  # No reintentar en el último intento
                # Restaurar el objeto cap en caso de error
                # Esto ayuda con formatos de video problemáticos
                if hasattr(self, 'video_path') and self.video_path:
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.video_path)
        
        # Si llegamos aquí, todos los intentos fallaron
        return None
    
    @profiler.track_time    
    def mask_helmet_yolo(self, color_image: np.ndarray, helmet_height_ratio: float = 0.3, prev_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Usa YOLOv8 para segmentar el casco y lo pinta de verde.
        Si se proporciona una máscara previa, la reutiliza.
        Args:
            color_image: Imagen en color (BGR).
            helmet_height_ratio: Proporción de la imagen a considerar como región del casco (parte inferior).
            prev_mask: Máscara previa para reutilizar (opcional).
        Returns:
            Tuple: (Imagen con la región del casco pintada de verde, Máscara generada o reutilizada).
        """
        # Copia de la imagen
        result_1 = color_image.copy()
        height, width = color_image.shape[:2]

        # Si hay una máscara previa, reutilizarla
        if prev_mask is not None:
            mask_final = prev_mask
        else:
            # Convertir la imagen a RGB (YOLOv8 espera imágenes en RGB)
            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Realizar la predicción con YOLOv8
            results = self.yolo_model(image_rgb, conf=0.2, iou=0.5,imgsz=224)  # Ajusta conf e iou según necesidad

            # Inicializar máscara vacía
            mask_final = np.zeros((height, width), dtype=np.uint8)

            # Procesar los resultados de segmentación
            if results[0].masks is not None:
                for result in results:
                    masks = result.masks.data.cpu().numpy()  # Máscaras de segmentación
                    boxes = result.boxes.xyxy.cpu().numpy()  # Cajas delimitadoras
                    classes = result.boxes.cls.cpu().numpy()  # Clases predichas

                    # Filtrar para la clase del casco (asumiendo que es la clase 0 o 'helmet')
                    # Si usas un modelo pre-entrenado en COCO, la clase 'helmet' no existe, usa 'person' (clase 0) y ROI
                    for i, cls in enumerate(classes):
                        # Ajusta según la clase de tu modelo. Ejemplo: clase 0 para 'helmet' en modelo personalizado
                        if int(cls) == 0:  # Cambia según el índice de clase de tu modelo
                            # Obtener la máscara correspondiente
                            '''mask = masks[i]
                            # Redimensionar la máscara al tamaño de la imagen
                            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                            mask = (mask > 0).astype(np.uint8) * 255  # Convertir a binario (0 o 255)

                            # Opcional: Filtrar usando la ROI inferior para enfocarse en el casco
                            roi_height = int(height * helmet_height_ratio)
                            roi_mask = np.zeros((height, width), dtype=np.uint8)
                            roi_mask[height - roi_height:, :] = 255  # Parte inferior
                            mask = cv2.bitwise_and(mask, roi_mask)

                            

                            # Combinar máscaras si hay múltiples detecciones
                            mask_final = cv2.bitwise_or(mask_final, mask)'''

                            mask = masks[i]
                            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                            mask = (mask > 0).astype(np.uint8) * 255
                            mask_final = cv2.bitwise_or(mask_final, mask)

                # Refinar la máscara con operaciones morfológicas
                kernel = np.ones((5, 5), np.uint8)
                mask_final = cv2.erode(mask_final, kernel, iterations=1)  # Eliminar ruido
                mask_final = cv2.dilate(mask_final, kernel, iterations=3)  # Expandir para cubrir el casco

            else:
                # Si no se detecta casco, devolver la imagen sin cambios y máscara vacía
                print("No helmet detected in this frame.")
                return result_1, mask_final

        # Crear una imagen verde del mismo tamaño que la imagen original
        green_color = np.zeros_like(color_image)  # Crear una imagen vacía
        green_color[:, :] = [125, 125, 125]  # Color verde en BGR (0, 255, 0)

        # Aplicar la máscara para pintar solo la región del casco
        masked_green = cv2.bitwise_and(green_color, green_color, mask=mask_final)

        # Crear máscara invertida para conservar el resto de la imagen
        mask_inv = cv2.bitwise_not(mask_final)

        # Combinar la región verde con el resto de la imagen original    
        
        result_original = cv2.bitwise_and(result_1, result_1, mask=mask_inv)
        result = cv2.add(masked_green, result_original)

        return result, mask_final
    
    def mask_helmet(self, img):
        """Mask the helmet region using SAM and paint it green."""
        print("Processing frame...")
        
        img = cv2.resize(img, (224, 224))
        height, width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        outputs = self.model.run(None, {"images":preprocess_image_tensor(img)})
        print("test")
        flag,result = postprocess_outputs(outputs, height, width)

        
        

        # Procesar los resultados de segmentación

        if flag is True:
            result_image = img.copy()
            overlay = np.zeros_like(img, dtype=np.uint8)
            color = (125, 125, 125, 255)  # RGBA color for the helmet
            # Extract RGB and alpha from color
            fill_color = color[:3]  # (R, G, B) = (125, 125, 125)
            alpha = color[3] / 255.0  # Normalize alpha to [0, 1]
            
            for obj in result:
                x1, y1, x2, y2, _, _, _, polygon = obj
                # Translate polygon coordinates relative to (x1, y1)
                polygon = [(round(x1 + point[0]), round(y1 + point[1])) for point in polygon]
                # Convert polygon to format required by cv2.fillPoly
                pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
                # Draw filled polygon on overlay
                cv2.fillPoly(overlay, [pts], fill_color)
            
            # Create alpha mask for blending
            mask = np.any(overlay != 0, axis=2).astype(np.float32)
            alpha_mask = mask * alpha

            for c in range(3):  # For each color channel
                result_image[:, :, c] = (1 - alpha_mask) * result_image[:, :, c] + alpha_mask * overlay[:, :, c]

            return result_image
        else:
            # Si no se detecta casco, devolver la imagen sin cambios y máscara vacía
            print("No helmet detected in this frame.")
            return img



        
        

    def extract_frames(self, start_frame: int, end_frame: int, fps_target: int = 10) -> List[np.ndarray]:
        """
        Extract frames con procesamiento vectorizado para mayor rendimiento, actualizando la máscara cada 10 frames.
        """
        frames, crude_frames = [], []

        # Calculate the total number of frames in the selection
        total_frames_selection = end_frame - start_frame + 1

        # Calculate the duration of the selection in seconds
        selection_duration = total_frames_selection / self.fps

        # Calculate total frames to extract based on target fps
        frames_to_extract = int(selection_duration * fps_target)
        frames_to_extract = max(1, frames_to_extract)

        # Vectorizar cálculo de índices
        if frames_to_extract < total_frames_selection:
            frame_indices = np.linspace(start_frame, end_frame, frames_to_extract, dtype=int)
        else:
            frame_indices = np.arange(start_frame, end_frame + 1)
        counter = 0
        # Procesamiento por lotes para reducir sobrecarga de función
        BATCH_SIZE =150
        last_mask = None  # Almacenar la última máscara generada

        for i in range(0, len(frame_indices), BATCH_SIZE):
            batch_indices = frame_indices[i:i+BATCH_SIZE]
            batch_frames = []
            

            # Extract the frames in the current batch
            for frame_num in batch_indices:
                frame = self.get_frame(frame_num)
                if frame is not None:
                    batch_frames.append((frame_num, frame))

            # Process the batch of frames
            if batch_frames:
                for idx, (frame_num, frame) in enumerate(batch_frames):
                    cropped = self.crop_frame(frame)

                    # Actualizar la máscara cada 3 frames
                    #if idx % 3 == 0 or last_mask is None:
                    '''result, last_mask = self.mask_helmet_yolo(
                        cropped,
                        helmet_height_ratio=self.helmet_height_ratio
                    )'''
                    '''else:
                        # Reutilizar la última máscara
                        result, _ = self.mask_helmet_yolo(
                            cropped,
                            helmet_height_ratio=self.helmet_height_ratio,
                            prev_mask=last_mask
                        )'''
                    result = self.mask_helmet(cropped)

                    clahe_image = self.apply_clahe(result)
                    #cv2.imwrite(f"img_test1/{str(i)}.png", clahe_image)
                    threshold_image = self.apply_treshold(clahe_image)
                    frames.append(threshold_image)

        return frames, crude_frames

    @profiler.track_time
    def crop_frame(self,image):

         
        if image is None:
            print(f"Error loading")
            return None
        
        height, width, _ = image.shape

        # Use the bottom half of the image
        #y_start = int(height * 0.53)
                   # 55% of the height
        y_start = int(height * self.y_start)  # 55% of the height
        crop_height = height - y_start         # height of bottom half
        square_size = crop_height              # base crop height

        # Increase width by 30%: new_width equals 130% of square_size
        new_width = square_size

        # Shift the crop center 20% to the right.
        # Calculate the desired center position.
        #x_center = int(width * 0.57)
        x_center = int(width * self.x_center)
        x_start = max(0, x_center - new_width // 2)
        x_end = x_start + new_width

        # Adapt the crop if x_end exceeds the image width
        if x_end > width:
            x_end = width
            x_start = max(0, width - new_width)

        # Crop the image: bottom half in height and new_width in horizontal dimension
        cropped_image = image[y_start:y_start+crop_height, x_start:x_end]

        
        print(cropped_image.shape)
        return cropped_image
    
    def crop_frame_example(self,image):

        if image is None:
            print(f"Error loading")
            return None
        
        height, width, _ = image.shape

        # Use the bottom half of the image
        #y_start = int(height * 0.53)
                   # 55% of the height
        y_start = int(height * self.y_start)  # 55% of the height
        crop_height = height - y_start         # height of bottom half
        square_size = crop_height              # base crop height

        # Increase width by 30%: new_width equals 130% of square_size
        new_width = square_size

        # Shift the crop center 20% to the right.
        # Calculate the desired center position.
        #x_center = int(width * 0.57)
        x_center = int(width * self.x_center)
        x_start = max(0, x_center - new_width // 2)
        x_end = x_start + new_width

        # Adapt the crop if x_end exceeds the image width
        if x_end > width:
            x_end = width
            x_start = max(0, width - new_width)

        # Crop the image: bottom half in height and new_width in horizontal dimension
        cropped_image = image[y_start:y_start+crop_height, x_start:x_end]
        cropped_image = recortar_imagen(cropped_image,self.starty, self.axes)
        cropped_image = recortar_imagen_again(cropped_image,self.starty, self.axes)
        #print(self.starty, self.axes, self.y_start, self.x_center)
        return cropped_image

    @profiler.track_time
    def apply_clahe(self, image):

        image = recortar_imagen(image,self.starty, self.axes)
        
        height, width = image.shape[:2]
        
        roi_mask = create_rectangular_roi(height, width, x1=int(width*.1), y1=int(height*.28), 
                                        x2=int(width*.9), y2=int(height*.95))
        
        # Aplicar CLAHE
        '''clahe_image = adaptive_clahe_iterative(
            image,
            roi_mask,
            initial_clip_limit=3.0,
            max_clip_limit=7.0,
            iterations=5,
            target_rms_min=0.18,
            target_rms_max=0.2,
            bright_threshold=200
        )'''

        clahe_image = cv2.createCLAHE(clipLimit=10, tileGridSize=(3, 3)).apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        #clahe_image = cv2.equalizeHist(image)
        return clahe_image
    
    @profiler.track_time
    def apply_treshold(self, image):

        #try:
        # Process the image with adaptive edge detection (target 6% de bordes)
        '''_, edges, _, config = adaptive_edge_detection(
            image, 
            min_edge_percentage=3,
            max_edge_percentage=6,
            target_percentage=5,
            max_attempts=5
        )'''
        percentage = calculate_black_pixels_percentage(image)
        _, edges, _, config = adaptive_edge_detection(
            image, 
            min_edge_percentage=percentage,
            max_edge_percentage=percentage,
            target_percentage=percentage,
            max_attempts=7
        )
        
        
        # Save the edge image
        if edges is not None:
            edges = recortar_imagen_again(edges,self.starty, self.axes)
            #edges = cv2.resize(edges,(224, 224))
            return edges
            
        '''except Exception as e:
            print(f"Error processing")
            return None'''
    
    def __del__(self):
        if self.cap is not None:
            self.cap.release()

