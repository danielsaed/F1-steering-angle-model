import cv2
import numpy as np
from typing import List, Tuple
import tempfile
import time
import functools
from collections import defaultdict
import onnxruntime as ort
from utils.model_handler import ModelHandler
from utils.helper import (
    preprocess_image_tensor,
    postprocess_outputs,
    recortar_imagen,
    recortar_imagen_again,
    calculate_black_pixels_percentage,
    adaptive_edge_detection,

    )
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from utils.helper import BASE_DIR

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


class VideoProcessor:
    def __init__(self):
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.target_fps = 10
        self.driver_crop_type = "albon"
        self.load_crop_variables(self.driver_crop_type)
        #self.yolo_model = YOLO("models/best.pt")
        self.model = ort.InferenceSession(Path(BASE_DIR) / "models" / "best-224.onnx")
        self.input_shape = (224, 224)  # Match imgsz=224 from your original code
        self.conf_thres = 0.5  # Confidence threshold
        self.iou_thres = 0.5   # IoU threshold for NMS


        self.frame_cache = OrderedDict()
        self.frame_cache_size = 50  # Reduced size to conserve memory
        self.last_position = -1
    
    def clear_cache(self):
        """Clear the frame cache to free memory."""
        self.frame_cache.clear()

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
        tfile = tempfile.NamedTemporaryFile(delete=True)
        tfile.write(video_file.read())
        
        # Guardar ruta para posibles reinicios
        self.video_path = tfile.name
        
        self.cap = cv2.VideoCapture(tfile.name)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"FPS: {self.fps}")
        print(f"Total frames: {self.total_frames}")
        
        return True
    import cv2


    def load_video2(self, video_file, output_resolution=(854, 480)) -> bool:
        """
        Load video file, resize to 480p, and get basic information.
        
        Args:
            video_file: Input video file object
            output_resolution: Tuple of (width, height) for resizing (default: 854x480 for 480p)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create temporary file to store the input video
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_file.read())
            tfile.close()  # Close the file to allow VideoCapture to access it

            # Store the temporary file path
            self.video_path = tfile.name

            # Load the video
            self.cap = cv2.VideoCapture(tfile.name)
            if not self.cap.isOpened():
                print("Error: Could not open video file.")
                return False

            # Get original video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            print(f"FPS: {self.fps}")
            print(f"Total frames: {self.total_frames}")

            # Prepare for resizing and saving to a new temporary file
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
            out = cv2.VideoWriter(output_path, fourcc, self.fps, output_resolution)

            # Process each frame
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                # Resize frame to 480p
                resized_frame = cv2.resize(frame, output_resolution, interpolation=cv2.INTER_AREA)
                out.write(resized_frame)

            # Release resources
            self.cap.release()
            out.release()

            # Update video path to the resized video
            self.video_path = output_path
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print("Error: Could not open resized video.")
                return False

            print(f"Video resized to {output_resolution} and saved to {output_path}")
            return True

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return False
    
    def load_video1(self, video_file) -> bool:
        """Load video file and get basic information"""
        with tempfile.TemporaryFile(suffix='.mp4') as tfile:
            tfile.write(video_file.read())
            tfile.seek(0)
            self.video_path = tfile.name  # Store for reference
            self.cap = cv2.VideoCapture(tfile.name)
            if not self.cap.isOpened():
                return False
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            return True
            
    @profiler.track_time
    def get_frame1(self, frame_number: int) -> np.ndarray:
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
    
    def get_frame(self, frame_number: int) -> np.ndarray:
        
        if self.cap is None:
            return None

        '''if frame_number in self.frame_cache:
            return self.frame_cache[frame_number]'''

        if hasattr(self, 'last_position') and frame_number == self.last_position + 1:
            ret, frame = self.cap.read()
            if ret:
                self.last_position = frame_number
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_cache[frame_number] = rgb_frame
                if len(self.frame_cache) > self.frame_cache_size:
                    self.frame_cache.popitem(last=False)  # Remove oldest item
                return cv2.resize(rgb_frame, (849, 477))

        for attempt in range(3):

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                self.last_position = frame_number
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_cache[frame_number] = rgb_frame
                if len(self.frame_cache) > self.frame_cache_size:
                    self.frame_cache.popitem(last=False)
                
                return cv2.resize(rgb_frame, (854,480), interpolation=cv2.INTER_LINEAR)

            if attempt < 2 and hasattr(self, 'video_path') and self.video_path:
                self.cap.release()
                self.cap = cv2.VideoCapture(self.video_path)
            
        print(f"Error reading frame {frame_number}, retrying...")
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
        print(f"Frame number: {frame_number}")
        
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
            try:
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
            except:
                pass
                
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
        
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
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

    def extract_frames1(self, start_frame: int, end_frame: int, fps_target: int = 10) -> List[np.ndarray]:
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

                    result = self.mask_helmet(cropped)

                    clahe_image = self.apply_clahe(result)

                    threshold_image = self.apply_treshold(clahe_image)

                    frames.append(threshold_image)

        return frames, crude_frames

    def extract_frames(self, start_frame: int, end_frame: int, fps_target: int = 10) -> List[np.ndarray]:
        frames, crude_frames = [], []

        total_frames_selection = end_frame - start_frame + 1
        selection_duration = total_frames_selection / self.fps
        frames_to_extract = max(1, int(selection_duration * fps_target))
        frame_indices = np.linspace(start_frame, end_frame, frames_to_extract, dtype=int) if frames_to_extract < total_frames_selection else np.arange(start_frame, end_frame + 1)

        BATCH_SIZE = 64

        def process_frame(frame_data):
            frame_num, frame = frame_data
            if frame is None:
                return None
            cropped = self.crop_frame(frame)
            result = self.mask_helmet(cropped)
            clahe_image = self.apply_clahe(result)
            threshold_image = self.apply_treshold(clahe_image)
            return threshold_image

        for i in range(0, len(frame_indices), BATCH_SIZE):
            batch_indices = frame_indices[i:i+BATCH_SIZE]
            batch_frames = [(idx, self.get_frame(idx)) for idx in batch_indices]
            with ThreadPoolExecutor(max_workers=2) as executor:  # Adjust max_workers based on CPU cores
                batch_results = list(executor.map(process_frame, [f for f in batch_frames if f[1] is not None]))
            frames.extend([r for r in batch_results if r is not None])

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
            max_attempts=1
        )
        
        # Save the edge image
        if edges is not None:
            edges = recortar_imagen_again(edges,self.starty, self.axes)
            return edges
            
    
    def __del__(self):
        if self.cap is not None:
            self.cap.release()
        self.clear_cache()  # Ensure cache is cleared on object deletion

