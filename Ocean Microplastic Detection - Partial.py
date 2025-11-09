# Ocean Microplastic Detection - Partial Project Source Code

This file contains all necessary Python code for training and deploying the microplastic detection system.

---

## File 1: preprocess.py - Image Preprocessing Module

```python
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import os

class MicroplasticPreprocessor:
    """Image preprocessing for microplastic detection"""
    
    def __init__(self, target_size=640):
        self.target_size = target_size
        
    def resize_image(self, image, maintain_aspect=True):
        """Resize image while maintaining aspect ratio"""
        h, w = image.shape[:2]
        
        if maintain_aspect:
            scale = self.target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(image, (new_w, new_h))
            
            # Pad to target size
            canvas = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            pad_y = (self.target_size - new_h) // 2
            pad_x = (self.target_size - new_w) // 2
            canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
            return canvas
        else:
            return cv2.resize(image, (self.target_size, self.target_size))
    
    def normalize_image(self, image, method='minmax'):
        """Normalize pixel values"""
        if method == 'minmax':
            return image.astype(np.float32) / 255.0
        elif method == 'zscore':
            mean = np.mean(image)
            std = np.std(image)
            return (image.astype(np.float32) - mean) / (std + 1e-8)
        return image
    
    def reduce_noise(self, image, method='gaussian'):
        """Apply noise reduction"""
        if method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        elif method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        return image
    
    def enhance_contrast(self, image):
        """Apply CLAHE for contrast enhancement"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel = image[:,:,0]
        else:
            l_channel = image
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_channel = clahe.apply(l_channel)
        
        if len(image.shape) == 3:
            image[:,:,0] = l_channel
            image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        return image
    
    def preprocess_pipeline(self, image_path, apply_all=True):
        """Complete preprocessing pipeline"""
        # Read image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing steps
        image = self.resize_image(image)
        image = self.reduce_noise(image, method='gaussian')
        image = self.enhance_contrast(image)
        image = self.normalize_image(image, method='minmax')
        
        return image


class DataAugmentation:
    """Data augmentation for training dataset"""
    
    def __init__(self):
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=25, p=0.7),
            A.RandomScale(scale=(0.8, 1.2), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        ])
    
    def augment_image(self, image):
        """Apply augmentation to single image"""
        augmented = self.transform(image=image)
        return augmented['image']
    
    def generate_augmented_dataset(self, input_dir, output_dir, num_augmentations=3):
        """Generate augmented dataset from original images"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file in os.listdir(input_dir):
            if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image = cv2.imread(os.path.join(input_dir, img_file))
                
                # Save original
                cv2.imwrite(str(output_dir / img_file), image)
                
                # Generate augmented versions
                for i in range(num_augmentations):
                    augmented = self.augment_image(image)
                    name, ext = os.path.splitext(img_file)
                    output_path = output_dir / f"{name}_aug_{i}{ext}"
                    cv2.imwrite(str(output_path), augmented)


# Example usage:
if __name__ == "__main__":
    preprocessor = MicroplasticPreprocessor(target_size=640)
    augmenter = DataAugmentation()
    
    # Preprocess single image
    processed_img = preprocessor.preprocess_pipeline('sample_image.jpg')
    
    # Generate augmented dataset
    augmenter.generate_augmented_dataset('original_images/', 'augmented_images/', num_augmentations=3)
```

---

## File 2: train.py - Model Training Script

```python
from ultralytics import YOLO
import torch
import os
from pathlib import Path

class MicroplasticDetectionTrainer:
    """Training wrapper for YOLOv8 microplastic detection"""
    
    def __init__(self, model_name='yolov8m', device=0):
        """
        Initialize trainer
        Args:
            model_name: YOLOv8 variant (n, s, m, l, x)
            device: GPU device ID
        """
        self.model_name = model_name
        self.device = device
        self.model = YOLO(f'{model_name}.pt')
        
    def check_gpu(self):
        """Verify GPU availability"""
        if torch.cuda.is_available():
            print(f"GPU Available: {torch.cuda.get_device_name(self.device)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.2f} GB")
        else:
            print("GPU not available. Using CPU.")
    
    def train(self, data_yaml, epochs=100, batch_size=16, imgsz=640, patience=20):
        """
        Train YOLOv8 model on custom dataset
        
        Args:
            data_yaml: Path to data.yaml file
            epochs: Number of training epochs
            batch_size: Batch size for training
            imgsz: Input image size
            patience: Early stopping patience
        """
        print(f"Starting training with {self.model_name}...")
        print(f"Dataset configuration: {data_yaml}")
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=self.device,
            patience=patience,
            save=True,
            project='runs/microplastic_detection',
            name='yolov8_microplastic',
            pretrained=True,
            optimizer='Adam',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            augment=True,
            verbose=True
        )
        
        print("Training completed!")
        return results
    
    def validate(self, model_path, data_yaml):
        """Validate trained model"""
        model = YOLO(model_path)
        metrics = model.val(data=data_yaml)
        return metrics
    
    def predict(self, model_path, source, conf=0.5):
        """Run inference on images or video"""
        model = YOLO(model_path)
        results = model.predict(source=source, conf=conf, save=True)
        return results


# Google Colab Training Setup
def setup_colab_training():
    """Setup Google Colab environment"""
    code = """
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install YOLOv8
!pip install ultralytics
!pip install -U albumentations

# Check GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Create data.yaml (replace paths with your dataset)
yaml_content = '''
path: /content/drive/MyDrive/microplastic_dataset
train: images/train
val: images/val
test: images/test

nc: 1
names: ['microplastic']
'''

with open('data.yaml', 'w') as f:
    f.write(yaml_content)

# Training code
from ultralytics import YOLO

model = YOLO('yolov8m.pt')
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    patience=20,
    save=True,
    project='runs/microplastic'
)

# View results
from IPython.display import Image
Image(filename='runs/microplastic/yolov8_microplastic/results.png')
    """
    return code


# Example usage:
if __name__ == "__main__":
    trainer = MicroplasticDetectionTrainer(model_name='yolov8m', device=0)
    trainer.check_gpu()
    
    # Train model
    results = trainer.train(
        data_yaml='data.yaml',
        epochs=100,
        batch_size=16,
        imgsz=640,
        patience=20
    )
    
    # Validate
    # metrics = trainer.validate('runs/microplastic_detection/yolov8_microplastic/weights/best.pt', 'data.yaml')
```

---

## File 3: inference.py - Inference & Deployment

```python
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
from datetime import datetime

class MicroplasticDetector:
    """Inference engine for microplastic detection"""
    
    def __init__(self, model_path, conf_threshold=0.5):
        """
        Initialize detector
        Args:
            model_path: Path to trained YOLOv8 weights
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
    
    def detect_image(self, image_path):
        """Detect microplastics in single image"""
        results = self.model(image_path, conf=self.conf_threshold)
        return results[0]
    
    def detect_batch(self, image_dir):
        """Detect microplastics in batch of images"""
        image_paths = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
        all_detections = []
        
        for img_path in image_paths:
            result = self.detect_image(str(img_path))
            all_detections.append({
                'image': str(img_path),
                'detections': result.boxes.data.cpu().numpy().tolist()
            })
        
        return all_detections
    
    def detect_video(self, video_path, output_path=None):
        """Detect microplastics in video stream"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if output_path:
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        frame_count = 0
        detections_log = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = self.model(frame, conf=self.conf_threshold)
            
            # Annotate frame
            annotated_frame = results[0].plot()
            
            # Log detections
            for box in results[0].boxes:
                detections_log.append({
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),
                    'conf': float(box.conf[0]),
                    'class': int(box.cls[0])
                })
            
            if output_path:
                writer.write(annotated_frame)
            
            cv2.imshow('Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if output_path:
            writer.release()
        cv2.destroyAllWindows()
        
        return detections_log
    
    def analyze_detections(self, detections):
        """Analyze detection statistics"""
        stats = {
            'total_detections': len(detections),
            'avg_confidence': np.mean([d['conf'] for d in detections]),
            'min_confidence': np.min([d['conf'] for d in detections]),
            'max_confidence': np.max([d['conf'] for d in detections]),
            'detection_times': {}
        }
        
        for d in detections:
            if d['timestamp'] not in stats['detection_times']:
                stats['detection_times'][d['timestamp']] = 0
            stats['detection_times'][d['timestamp']] += 1
        
        return stats