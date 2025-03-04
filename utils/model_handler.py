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
class ModelHandler:
    def __init__(self):
        # Placeholder for actual model loading
        self.current_model = None
        self.available_models = {
            "F1 Steering Angle Detection": r"models/f1-steering-angle-model.onnx",
            "Track Position Analysis": "position_model",
            "Driver Behavior Analysis": "behavior_model"
        }
        
    def process_frames(self, frames: List[np.ndarray], model_name: str) -> Dict:
        """Process frames through selected model"""
        # Placeholder for actual model inference
        # In reality, this would use a proper ML model

        self.current_model = ort.InferenceSession(self.available_models[model_name])

        results = []

        for i, frame in enumerate(frames):

            input_data = preprocess_image_exactly_like_pytorch(frame)
            input_name = self.current_model.get_inputs()[0].name
        
            # Ejecutar la inferencia
            ort_inputs = {input_name: input_data}
            ort_outputs = self.current_model.run(None, ort_inputs)
            
            # Procesar resultado
            predicted_angle_normalized = ort_outputs[0][0][0]
            
            # Desnormalizar el ángulo
            angle = denormalize_angles(predicted_angle_normalized)

            confidence = np.random.uniform(0.7, 0.99)
            results.append({
                'frame_number': i,
                'steering_angle': angle,
                'confidence': confidence
            })


        return results
        
    def export_results(self, results: Dict) -> pd.DataFrame:
        """Convert results to pandas DataFrame for export"""
        df = pd.DataFrame(results)
        return df
