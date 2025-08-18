import torch
import torch.nn as nn
import numpy as np
import cv2
from ultralytics import YOLO
from ..core import FaceDetector
from ...utils import *


class YOLOv8FaceDetector(FaceDetector):
    def __init__(self, device='cuda', path_to_detector=None, verbose=False, **kwargs):
        super(YOLOv8FaceDetector, self).__init__(device, verbose)
        
        # Initialize YOLOv8 model
        if path_to_detector is None:
            # Use pre-trained YOLOv8 model, will be fine-tuned for faces
            self.face_detector = YOLO('yolov8n.pt')  # nano version for speed
        else:
            self.face_detector = YOLO(path_to_detector)
            
        # Move model to device
        if device == 'cuda' and torch.cuda.is_available():
            self.face_detector.to(device)
            
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.5)
        self.nms_threshold = kwargs.get('nms_threshold', 0.4)

    def detect_from_image(self, tensor_or_path):
        if isinstance(tensor_or_path, str):
            # Load image from path
            image = cv2.imread(tensor_or_path)
            if image is None:
                return []
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Convert tensor to numpy array
            if isinstance(tensor_or_path, torch.Tensor):
                if tensor_or_path.dim() == 4:
                    tensor_or_path = tensor_or_path.squeeze(0)
                image = tensor_or_path.permute(1, 2, 0).cpu().numpy()
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
            else:
                image = tensor_or_path
                
        # Ensure image is in RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Run YOLOv8 detection
            results = self.face_detector.predict(
                image, 
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                classes=[0],  # Only detect 'person' class initially
                verbose=False
            )
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confidences):
                        if conf >= self.confidence_threshold:
                            # Filter boxes that are likely faces based on aspect ratio
                            x1, y1, x2, y2 = box
                            width = x2 - x1
                            height = y2 - y1
                            aspect_ratio = width / height if height > 0 else 0
                            
                            # Face aspect ratio is typically between 0.7 and 1.3
                            if 0.5 <= aspect_ratio <= 2.0:
                                # Check if detection is in upper portion of image (faces usually in upper half)
                                img_height = image.shape[0]
                                center_y = (y1 + y2) / 2
                                if center_y < img_height * 0.8:  # Allow some flexibility
                                    detections.append([x1, y1, x2, y2, conf])
            
            return detections
        else:
            return []

    def detect_from_batch(self, tensor):
        """Detect faces from a batch of images"""
        batch_detections = []
        
        if isinstance(tensor, torch.Tensor):
            if tensor.dim() == 4:  # Batch of images
                batch_size = tensor.shape[0]
                for i in range(batch_size):
                    single_image = tensor[i]
                    detections = self.detect_from_image(single_image)
                    batch_detections.append(detections)
            else:
                # Single image
                detections = self.detect_from_image(tensor)
                batch_detections.append(detections)
        else:
            # List of images
            for image in tensor:
                detections = self.detect_from_image(image)
                batch_detections.append(detections)
                
        return batch_detections

    @property
    def reference_scale(self):
        return 195

    @property 
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0


class YOLOv8FaceDetectorFinetuned(YOLOv8FaceDetector):
    """YOLOv8 detector specifically fine-tuned for face detection"""
    
    def __init__(self, device='cuda', path_to_detector=None, verbose=False, **kwargs):
        # Use a face-specific model if available
        if path_to_detector is None:
            # Try to use YOLOv8 face model (would need to be trained/downloaded)
            path_to_detector = 'yolov8n-face.pt'  # hypothetical face model
            
        super(YOLOv8FaceDetectorFinetuned, self).__init__(
            device=device, 
            path_to_detector=path_to_detector, 
            verbose=verbose, 
            **kwargs
        )
        
        # More restrictive parameters for face detection
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.6)
        self.nms_threshold = kwargs.get('nms_threshold', 0.3)

    def detect_from_image(self, tensor_or_path):
        if isinstance(tensor_or_path, str):
            image = cv2.imread(tensor_or_path)
            if image is None:
                return []
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            if isinstance(tensor_or_path, torch.Tensor):
                if tensor_or_path.dim() == 4:
                    tensor_or_path = tensor_or_path.squeeze(0)
                image = tensor_or_path.permute(1, 2, 0).cpu().numpy()
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
            else:
                image = tensor_or_path

        # Run YOLOv8 detection with face-specific classes if available
        try:
            results = self.face_detector.predict(
                image,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                verbose=False
            )
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confidences):
                        if conf >= self.confidence_threshold:
                            x1, y1, x2, y2 = box
                            detections.append([x1, y1, x2, y2, conf])
            
            return detections
            
        except Exception as e:
            if self.verbose:
                print(f"YOLOv8 face detection failed: {e}")
            # Fallback to generic person detection with face filtering
            return super().detect_from_image(tensor_or_path)