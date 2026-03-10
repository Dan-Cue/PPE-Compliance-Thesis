# classifier.py
"""
TFLite PPE Classifier for Raspberry Pi 5 Deployment
Supports both float16 and int8 quantized models for maximum performance
"""

import numpy as np
import cv2

class PPEClassifier:
    def __init__(self, model_path):
        """
        Load TFLite classification model for PPE verification.
        Optimized for Raspberry Pi 5 deployment.
        
        Args:
            model_path: Path to .tflite model file
        """
        print(f"Loading TFLite classifier from: {model_path}")
        
        try:
            import tensorflow as tf
            
            # Load TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Get input shape
            self.input_shape = self.input_details[0]['shape']
            self.input_height = self.input_shape[1]
            self.input_width = self.input_shape[2]
            
            # Check if model is quantized
            self.input_dtype = self.input_details[0]['dtype']
            self.is_quantized = self.input_dtype == np.uint8
            
            # Get quantization parameters if quantized
            if self.is_quantized:
                self.input_scale = self.input_details[0]['quantization'][0]
                self.input_zero_point = self.input_details[0]['quantization'][1]
                self.output_scale = self.output_details[0]['quantization'][0]
                self.output_zero_point = self.output_details[0]['quantization'][1]
                print(f"  ✓ Quantized INT8 model detected")
            else:
                print(f"  ✓ Float model detected")
            
            print(f"  Input shape: {self.input_height}x{self.input_width}")
            print(f"  Input dtype: {self.input_dtype}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load TFLite model: {e}")
        
        # Class names in order (must match training order)
        self.class_names = [
            "apron",
            "boots", 
            "gloves",
            "haircap",
            "long_sleeves",
            "mask"
        ]
        
        print(f"✓ TFLite Classifier loaded successfully!")
        print(f"  Output classes: {len(self.class_names)}")
    
    def preprocess_image(self, image):
        """
        Preprocess image for TFLite model.
        
        Args:
            image: BGR image (numpy array)
            
        Returns:
            Preprocessed image ready for model input
        """
        # Resize to model input size
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        if self.is_quantized:
            # For INT8 quantized model
            # Input should be uint8 in range [0, 255]
            preprocessed = rgb.astype(np.uint8)
        else:
            # For float models (float32 or float16)
            # Normalize to [0, 1]
            preprocessed = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(preprocessed, axis=0)
        
        return batched
    
    def classify(self, image, expected_class=None, confidence_threshold=0.7):
        """
        Classify a cropped PPE region using TFLite.
        
        Args:
            image: Cropped PPE region (BGR)
            expected_class: Expected class name (e.g., "mask")
            confidence_threshold: Minimum confidence to consider valid
            
        Returns:
            dict with keys:
                - predicted_class: str
                - confidence: float
                - is_correct: bool (True if matches expected_class)
                - all_confidences: dict of all class confidences
        """
        # Preprocess
        preprocessed = self.preprocess_image(image)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensor
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Dequantize if needed
        if self.is_quantized:
            # Convert from uint8 to float
            output = (output.astype(np.float32) - self.output_zero_point) * self.output_scale
        
        # Apply softmax if needed (TFLite models sometimes don't include softmax)
        # Check if output is already probabilities (sum ~1.0)
        if not (0.99 <= np.sum(output) <= 1.01):
            # Apply softmax
            exp_output = np.exp(output - np.max(output))
            predictions = exp_output / np.sum(exp_output)
        else:
            predictions = output
        
        # Get top prediction
        predicted_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        # Build confidence dict
        all_confidences = {
            self.class_names[i]: float(predictions[i]) 
            for i in range(len(self.class_names))
        }
        
        # Check if correct
        is_correct = False
        if expected_class is not None:
            is_correct = (
                predicted_class == expected_class and 
                confidence >= confidence_threshold
            )
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "is_correct": is_correct,
            "all_confidences": all_confidences
        }
    
    def verify_detection(self, frame, bbox, expected_class, confidence_threshold=0.7):
        """
        Verify a YOLO detection using TFLite classification.
        
        Args:
            frame: Full frame image
            bbox: Bounding box (x1, y1, x2, y2)
            expected_class: Expected PPE class
            confidence_threshold: Minimum confidence
            
        Returns:
            Classification result dict
        """
        x1, y1, x2, y2 = bbox
        
        # Add some padding to bbox
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        # Crop region
        cropped = frame[y1:y2, x1:x2]
        
        # Handle empty crop
        if cropped.size == 0:
            return {
                "predicted_class": "unknown",
                "confidence": 0.0,
                "is_correct": False,
                "all_confidences": {}
            }
        
        # Classify
        return self.classify(cropped, expected_class, confidence_threshold)
