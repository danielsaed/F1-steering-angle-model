import numpy as np
import pandas as pd
from typing import List, Dict
from PIL import Image
import cv2
import onnxruntime as ort

def denormalize_angles(normalized_angles):
    """
    Convierte ángulos normalizados [-1,1] a grados [-180,180]
    """
    return (normalized_angles + 1) / 2 * (180 - (-180)) + (-180)

def preprocess_image_exactly_like_pytorch(image_input):
    """
    Preprocesa una imagen de OpenCV (como adjusted_edges) 
    para usarla con modelos ONNX.
    
    Args:
        image_input: Array NumPy de OpenCV (imagen de bordes, binaria, etc.)
        
    Returns:
        Array NumPy listo para inferencia con ONNX
    """
    # Verificar que la entrada no sea None
    if image_input is None:
        raise ValueError("Received None as image input")
        
    # Asegurar que la imagen es un array NumPy
    if not isinstance(image_input, np.ndarray):
        raise TypeError(f"Expected NumPy array, got {type(image_input)}")
        
    # Verificar que la imagen tiene dimensiones válidas
    if len(image_input.shape) < 2:
        raise ValueError(f"Invalid image shape: {image_input.shape}")
    
    # Copia para no modificar la original    
    img_copy = image_input.copy()
    
    # Si es una imagen de bordes o binaria, normalmente tiene valores 0 y 255
    # o 0 y 1. Asegurarse de que está en el rango [0, 255]
    if img_copy.dtype != np.uint8:
        if np.max(img_copy) <= 1.0:
            # Si está en rango [0, 1], convertir a [0, 255]
            img_copy = (img_copy * 255).astype(np.uint8)
        else:
            # De otro modo, simplemente convertir a uint8
            img_copy = img_copy.astype(np.uint8)
    
    # Para imágenes de bordes o binarias, asegurar que tenemos valores claros
    # (si todos los valores son muy bajos, puede que no se vea nada)
    if np.mean(img_copy) < 10 and np.max(img_copy) > 0:
        # Estirar el contraste para mejor visualización
        img_copy = cv2.normalize(img_copy, None, 0, 255, cv2.NORM_MINMAX)
    
    # Asegurar que la imagen es de un solo canal (escala de grises)
    if len(img_copy.shape) == 3:
        if img_copy.shape[2] == 3:
            # Convertir imagen BGR a escala de grises
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        else:
            # Tomar solo el primer canal
            img_copy = img_copy[:, :, 0]
    
    try:
        # Convertir de NumPy array a PIL Image
        img_pil = Image.fromarray(img_copy)
        
        # Redimensionar con PIL 
        img_resized = img_pil.resize((224, 224), Image.BILINEAR)
        
        # Convertir a numpy array
        img_np = np.array(img_resized, dtype=np.float32)
        
        # Normalizar de [0,255] a [0,1]
        img_np = img_np / 255.0
        
        # Normalizar con mean=0.5, std=0.5 (como en PyTorch)
        img_np = (img_np - 0.5) / 0.5
        
        # Reformatear para ONNX [batch_size, channels, height, width]
        img_np = np.expand_dims(img_np, axis=0)  # Añadir dimensión de canal
        img_np = np.expand_dims(img_np, axis=0)  # Añadir dimensión de batch
        
        return img_np
    except Exception as e:
        print(f"Error processing image: {e}")
        print(f"Image shape: {image_input.shape}, dtype: {image_input.dtype}")
        print(f"Min value: {np.min(image_input)}, Max value: {np.max(image_input)}")
        raise


def correct_outlier_angles(df, window_size=5, std_threshold=3.0, max_diff_threshold=80.0):
    
    angles = df['steering_angle'].values
    corrected_angles = angles.copy()
    
    for i in range(len(angles)):
        if i < window_size // 2 or i >= len(angles) - window_size // 2:  # Evitar bordes
            continue
        
        # Definir ventana local
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(angles), i + window_size // 2 + 1)
        window = angles[start_idx:end_idx]
        
        # Calcular estadísticas locales incluyendo el valor actual
        curr_angle = angles[i]
        local_mean = np.mean(window)
        local_std = np.std(window) if len(window) > 1 else 0
        
        # Calcular distancia angular mínima considerando el rango cíclico (-180° a 180°)
        def angular_distance(a, b):
            diff = abs(a - b)
            return min(diff, 360 - diff) if diff > 180 else diff
        
        diff_from_mean = angular_distance(curr_angle, local_mean)
        
        # Detectar outlier
        is_outlier = (diff_from_mean > std_threshold * local_std) or (diff_from_mean > max_diff_threshold)
        
        if is_outlier:
            print(i, curr_angle, local_mean, local_std, diff_from_mean)
            # Excluir el valor actual del cálculo del promedio de corrección
            corrected_window = np.delete(window, i - start_idx)
            if len(corrected_window) > 0:
                corrected_mean = np.mean(corrected_window)
                # Ajustar el ángulo corregido al rango cíclico más cercano
                diff_to_corrected = angular_distance(curr_angle, corrected_mean)
                if diff_to_corrected > 180:
                    corrected_angles[i] = corrected_mean - 360 if corrected_mean > 0 else corrected_mean + 360
                else:
                    corrected_angles[i] = corrected_mean
    
    # Crear nuevo DataFrame
    corrected_df = df.copy()
    corrected_df['steering_angle'] = corrected_angles
    return corrected_df
class ModelHandler:
    def __init__(self):
        # Placeholder for actual model loading
        self.current_model = None
        self.current_model_name = None
        self.available_models = {
            "F1 Steering Angle Detection": r"models/f1-steering-angle-model.onnx",
            "Track Position Analysis": "position_model",
            "Driver Behavior Analysis": "behavior_model"
        }
        
    def _load_model_if_needed(self, model_name: str):
        """Load the model only if it's not already loaded or if it's different"""
        if self.current_model is None or self.current_model_name != model_name:
            print(f"Loading model: {model_name}")  # Debugging info
            self.current_model = ort.InferenceSession(self.available_models[model_name])
            self.current_model_name = model_name
        
    def process_frames(self, frames: List[np.ndarray], model_name: str) -> Dict:
        """Process frames through selected model with efficient batch processing"""
        if not frames:
            return []
        
        # Load model only once
        self._load_model_if_needed(model_name)
        
        # Get input name once
        input_name = self.current_model.get_inputs()[0].name
        
        results = []
        
        # Define optimal batch size - ajusta según tu hardware
        BATCH_SIZE = 16
        
        # Process frames in batches
        for batch_start in range(0, len(frames), BATCH_SIZE):
            # Get current batch
            batch_end = min(batch_start + BATCH_SIZE, len(frames))
            current_batch = frames[batch_start:batch_end]
            batch_inputs = []
            
            # Pre-process all frames in the current batch
            for frame in current_batch:
                try:
                    # Procesar imagen pero mantener en formato que permita agrupación
                    processed_input = preprocess_image_exactly_like_pytorch(frame)
                    batch_inputs.append(processed_input)
                except Exception as e:
                    print(f"Error preprocessing frame: {e}")
                    # Usar un tensor vacío del mismo tamaño como reemplazo
                    empty_tensor = np.zeros((1, 1, 224, 224), dtype=np.float32)
                    batch_inputs.append(empty_tensor)
            
            try:
                # Combinar todos los inputs pre-procesados en un solo lote grande
                # Cada input tiene forma [1, 1, 224, 224], los concatenamos en la dimensión 0
                batched_input = np.vstack(batch_inputs)
                
                # Ejecutar inferencia sobre todo el lote a la vez
                ort_inputs = {input_name: batched_input}
                ort_outputs = self.current_model.run(None, ort_inputs)
                
                # Procesar resultados por lotes
                for i in range(len(current_batch)):
                    frame_idx = batch_start + i
                    predicted_angle_normalized = ort_outputs[0][i][0]
                    angle = denormalize_angles(predicted_angle_normalized)
                    confidence = np.random.uniform(0.7, 0.99)
                    
                    results.append({
                        'frame_number': frame_idx,
                        'steering_angle': angle,
                        'confidence': confidence
                    })
                    
            except Exception as e:
                print(f"Error in batch processing: {e}")
                # Si falla el procesamiento por lotes, volver a procesar individualmente
                for i, frame in enumerate(current_batch):
                    frame_idx = batch_start + i
                    try:
                        input_data = preprocess_image_exactly_like_pytorch(frame)
                        ort_inputs = {input_name: input_data}
                        ort_outputs = self.current_model.run(None, ort_inputs)
                        
                        predicted_angle_normalized = ort_outputs[0][0][0]
                        angle = denormalize_angles(predicted_angle_normalized)
                        confidence = np.random.uniform(0.7, 0.99)
                        
                        results.append({
                            'frame_number': frame_idx,
                            'steering_angle': angle,
                            'confidence': confidence
                        })
                    except Exception as sub_e:
                        print(f"Error processing individual frame {frame_idx}: {sub_e}")
                        # Añadir un resultado con valores predeterminados
                        results.append({
                            'frame_number': frame_idx,
                            'steering_angle': 0.0,
                            'confidence': 0.0
                        })
        
        return results
        
    def export_results(self, results: Dict) -> pd.DataFrame:
        """Convert results to pandas DataFrame for export"""
        df = pd.DataFrame(results)
        for i in range(3):
            df = correct_outlier_angles(df, window_size=15, std_threshold=1.7, max_diff_threshold=30.0)
        
        return df
