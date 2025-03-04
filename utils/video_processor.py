import cv2
import numpy as np
from typing import List
import tempfile
import os

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
def recortar_imagen(image):
    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    start_y = int(0.38 * height)
    cv2.rectangle(mask, (0, start_y), (width, height), 255, -1)
    center = (width // 2, start_y)
    axes = (width // 2, int(0.23 * height))
    cv2.ellipse(mask, center, axes, 0, 180, 360, 255, -1)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def recortar_imagen_again(image):
    
    height, width = image.shape

    mask = np.zeros((height, width), dtype=np.uint8)

    start_y = int(0.41 * height)
    cv2.rectangle(mask, (0, start_y), (width, height), 255, -1)
    center = (width // 2, start_y)
    axes = (width // 2, int(0.23 * height))
    cv2.ellipse(mask, center, axes, 0, 180, 360, 255, -1)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

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
    
    preprocessed_image = preprocess_image(original_gray)
    
    best_image = preprocessed_image.copy()
    best_rms = calculate_robust_rms_contrast(preprocessed_image, roi_mask, bright_threshold)
    clip_limit = initial_clip_limit
    
    for i in range(iterations):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(16, 16))
        current_image = clahe.apply(preprocessed_image)
        
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

def adaptive_edge_detection(imagen, min_edge_percentage=5.5, max_edge_percentage=6.5, target_percentage=6.0, max_attempts=5):
    """
    Detecta bordes con ajuste progresivo de parámetros hasta lograr un porcentaje óptimo
    de píxeles de borde en la imagen (entre min_edge_percentage y max_edge_percentage).
    
    Args:
        image_path: Ruta a la imagen
        min_edge_percentage: Porcentaje mínimo de píxeles que deben ser detectados como bordes (0-100)
        max_edge_percentage: Porcentaje máximo de píxeles que deben ser detectados como bordes (0-100)
        target_percentage: Porcentaje ideal de bordes a obtener
        max_attempts: Número máximo de intentos de ajuste
        show_results: Mostrar gráficos de resultado
    
    Returns:
        enhanced: Imagen mejorada
        best_edges: Mejores bordes detectados
        original: Imagen original
    """
    # Read image
    original = imagen
    if original is None:
        print(f"Error loading image")
        return None, None, None, None
    
    # Convert to grayscale
    #gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray = original
    
    # Calculate total pixels for percentage calculation
    total_pixels = gray.shape[0] * gray.shape[1]
    min_edge_pixels = int((min_edge_percentage / 100) * total_pixels)
    max_edge_pixels = int((max_edge_percentage / 100) * total_pixels)
    target_edge_pixels = int((target_percentage / 100) * total_pixels)
    
    # Initial parameters - ajustados para conseguir un rango alrededor del 6% de bordes
    clip_limits = [1.5, 2.0, 2.5, 3.0, 3.5]
    grid_sizes = [(8, 8), (8, 8), (8, 8), (8, 8), (8, 8)]
    # Empezamos con umbrales más altos para restringir la cantidad de bordes
    canny_thresholds = [(75, 180), (65, 170), (55, 160), (45, 150), (35, 140)]
    
    best_edges = None
    best_enhanced = None
    best_config = None
    best_edge_score = float('inf')  # Inicializamos con un valor alto
    
    # Try progressively more aggressive parameters
    for attempt in range(max_attempts):
        # Get parameters for this attempt
        clip_limit = clip_limits[attempt]
        grid_size = grid_sizes[attempt]
        low_threshold, high_threshold = canny_thresholds[attempt]
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        enhanced = clahe.apply(gray)
        
        # Apply noise reduction for higher attempts
        if attempt >= 2:
            enhanced = cv2.bilateralFilter(enhanced, 5, 100, 100)
        
        # Edge detection
        edges = cv2.Canny(enhanced, low_threshold, high_threshold)
        std_intensity = np.std(edges)
        
        # Reducir ruido con operaciones morfológicas
        kernel = np.ones((1, 1), np.uint8)
        edges = cv2.morphologyEx(
            edges, 
            cv2.MORPH_OPEN, 
            kernel, 
            iterations=0 if std_intensity < 60 else 1  # Más iteraciones si hay más ruido
        )
        
        # Count edge pixels
        edge_count = np.count_nonzero(edges)
        edge_percentage = (edge_count / total_pixels) * 100
        #print(f"Attempt {attempt + 1}: {edge_count} edge pixels ({edge_percentage:.2f}%)")
        
        # Ajuste fino para alcanzar el rango de 5.5-6.5%
        if edge_count > max_edge_pixels:  # Si hay demasiados bordes (>6.5%)
            # Ajustar el umbral de Canny para reducir la cantidad de bordes
            for threshold_adjustment in range(12):  # Más ajustes para mayor precisión
                # Aumentar ambos umbrales en 5 unidades (más granular)
                new_low = low_threshold + 5 * (threshold_adjustment + 1)
                new_high = high_threshold + 5 * (threshold_adjustment + 1)
                
                # Volver a aplicar Canny con umbrales ajustados
                adjusted_edges = cv2.Canny(enhanced, new_low, new_high)
                adjusted_edges = cv2.morphologyEx(
                    adjusted_edges, 
                    cv2.MORPH_OPEN, 
                    kernel, 
                    iterations=0 if std_intensity < 60 else 1
                )
                
                # Contar bordes nuevamente
                adjusted_count = np.count_nonzero(adjusted_edges)
                adjusted_percentage = (adjusted_count / total_pixels) * 100
                
                #print(f"  Adjustment +: {adjusted_count} edge pixels ({adjusted_percentage:.2f}%)")
                
                # Si ahora estamos en el rango deseado, usar estos bordes
                if adjusted_count <= max_edge_pixels:
                    edges = adjusted_edges
                    edge_count = adjusted_count
                    edge_percentage = adjusted_percentage
                    break
        elif edge_count < min_edge_pixels:  # Si hay muy pocos bordes (<5.5%)
            # Ajustar el umbral de Canny para aumentar la cantidad de bordes
            for threshold_adjustment in range(12):  # Más ajustes para mayor precisión
                # Disminuir ambos umbrales en 5 unidades (más granular)
                new_low = max(10, low_threshold - 5 * (threshold_adjustment + 1))
                new_high = max(30, high_threshold - 5 * (threshold_adjustment + 1))
                
                # Volver a aplicar Canny con umbrales ajustados
                adjusted_edges = cv2.Canny(enhanced, new_low, new_high)
                adjusted_edges = cv2.morphologyEx(
                    adjusted_edges, 
                    cv2.MORPH_OPEN, 
                    kernel, 
                    iterations=0 if std_intensity < 60 else 1
                )
                
                # Contar bordes nuevamente
                adjusted_count = np.count_nonzero(adjusted_edges)
                adjusted_percentage = (adjusted_count / total_pixels) * 100
                
                #print(f"  Adjustment -: {adjusted_count} edge pixels ({adjusted_percentage:.2f}%)")
                
                # Si ahora estamos en el rango deseado, usar estos bordes
                if adjusted_count >= min_edge_pixels and adjusted_count <= max_edge_pixels:
                    edges = adjusted_edges
                    edge_count = adjusted_count
                    edge_percentage = adjusted_percentage
                    break
        
        # Calculate how close we are to the target percentage (using absolute difference)
        edge_score = abs(edge_count - target_edge_pixels)
        
        # Record the best attempt (closest to target percentage)
        if edge_score < best_edge_score:
            best_edge_score = edge_score
            best_edges = edges
            best_enhanced = enhanced
            best_config = {
                'attempt': attempt + 1,
                'clip_limit': clip_limit,
                'grid_size': grid_size,
                'canny_thresholds': (low_threshold, high_threshold),
                'edge_pixels': edge_count,
                'edge_percentage': edge_percentage
            }
            #print(f"New best configuration: {edge_percentage:.2f}% edges (target: {target_percentage:.1f}%)")
        
        # If we are close enough to the target percentage, we can stop
        if abs(edge_percentage - target_percentage) < 0.2:  # Within 0.2% of target
            break
    
    return best_enhanced, best_edges, original, best_config



class VideoProcessor:
    def __init__(self):
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.target_fps = 10
        
    def load_video(self, video_file) -> bool:

        """Load video file and get basic information"""
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        self.cap = cv2.VideoCapture(tfile.name)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        return True
            
        
    def get_frame(self, frame_number: int) -> np.ndarray:
        """Get specific frame from video"""
        if self.cap is None:
            return None
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
        
    def extract_frames(self, start_frame: int, end_frame: int, fps_target: int = 10) -> List[np.ndarray]:
        """
        Extract frames between start and end points at specified frames per second
        
        Args:
            start_frame: First frame to extract
            end_frame: Last frame to extract
            fps_target: Target frames per second to extract (default: 10)
            
        Returns:
            List of extracted frames
        """
        frames,crude_frames = [],[]
        
        # Calculate the total number of frames in the selection
        total_frames_selection = end_frame - start_frame + 1
        
        # Calculate the duration of the selection in seconds
        selection_duration = total_frames_selection / self.fps
        
        # Calculate total frames to extract based on target fps
        frames_to_extract = int(selection_duration * fps_target)
        
        # Ensure we extract at least one frame
        frames_to_extract = max(1, frames_to_extract)
        
        # If we want fewer frames than the selection has
        if frames_to_extract < total_frames_selection:
            # Calculate indices for evenly distributed frames
            indices = np.linspace(start_frame, end_frame, frames_to_extract, dtype=int)
            frame_indices = indices
        else:
            # If we want all frames or more (limited to what's available)
            frame_indices = range(start_frame, end_frame + 1)
        
        # Extract the frames at the calculated indices
        for frame_num in frame_indices:
            frame = self.get_frame(frame_num)
            
            if frame is not None:
                crude_frames.append(frame)
                frame = self.crop_frame(frame)
                frame = self.apply_clahe(frame)
                frame = self.apply_treshold(frame)
                frames.append(frame)
        
        return frames,crude_frames

    def crop_frame(self,image):

        if image is None:
            print(f"Error loading")
            return None
        
        height, width, _ = image.shape

        # Use the bottom half of the image
        y_start = int(height * 0.53)           # 55% of the height
        crop_height = height - y_start         # height of bottom half
        square_size = crop_height              # base crop height

        # Increase width by 30%: new_width equals 130% of square_size
        new_width = square_size

        # Shift the crop center 20% to the right.
        # Calculate the desired center position.
        x_center = int(width * 0.57)
        x_start = max(0, x_center - new_width // 2)
        x_end = x_start + new_width

        # Adapt the crop if x_end exceeds the image width
        if x_end > width:
            x_end = width
            x_start = max(0, width - new_width)

        # Crop the image: bottom half in height and new_width in horizontal dimension
        cropped_image = image[y_start:y_start+crop_height, x_start:x_end]
        
        return cropped_image

    def apply_clahe(self, image):

        image = recortar_imagen(image)
        height, width = image.shape[:2]
        roi_mask = create_rectangular_roi(height, width, x1=int(width*.1), y1=int(height*.28), 
                                        x2=int(width*.9), y2=int(height*.95))
        
        # Aplicar CLAHE
        clahe_image = adaptive_clahe_iterative(
            image,
            roi_mask,
            initial_clip_limit=1.0,
            max_clip_limit=8.0,
            iterations=50,
            target_rms_min=0.17,
            target_rms_max=0.5,
            bright_threshold=220
        )

        return clahe_image
    
    def apply_treshold(self, image):

        #try:
        # Process the image with adaptive edge detection (target 6% de bordes)
        _, edges, _, config = adaptive_edge_detection(
            image, 
            min_edge_percentage=5.5,
            max_edge_percentage=6.5,
            target_percentage=6.0,
            max_attempts=5
        )
        
        # Save the edge image
        if edges is not None:
            edges = recortar_imagen_again(edges)
            return edges
            
        '''except Exception as e:
            print(f"Error processing")
            return None'''
            

        
    def __del__(self):
        if self.cap is not None:
            self.cap.release()
