import cv2
import torch
import numpy as np
import time
import threading
from PIL import Image
from collections import deque

class InferenceStats:
    """Tracks real-time inference statistics for the dashboard."""
    def __init__(self, window_size=30):
        self.lock = threading.Lock()
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.confidences = deque(maxlen=100)
        self.face_count = 0
        self.last_frame_time = time.time()
    
    def record_frame(self, latency_ms, confidences, face_count):
        with self.lock:
            now = time.time()
            self.frame_times.append(now - self.last_frame_time)
            self.last_frame_time = now
            self.latencies.append(latency_ms)
            self.confidences.extend(confidences)
            self.face_count = face_count
    
    def get_stats(self):
        with self.lock:
            fps = 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0
            avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
            
            # Build confidence histogram (10 bins: 0-10, 10-20, ..., 90-100)
            histogram = [0] * 10
            for c in self.confidences:
                bin_idx = min(int(c * 10), 9)
                histogram[bin_idx] += 1
            
            return {
                'fps': fps,
                'latency': avg_latency,
                'face_count': self.face_count,
                'confidence_histogram': histogram
            }

class VideoCamera:
    def __init__(self, model_manager):
        self.video = None  # Start with no video source
        self.lock = threading.Lock()
        self.model_manager = model_manager
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.current_model = None
        self.current_model_type = None
        self.current_backbone = None
        self.current_scaler = None
        self.transform = self.model_manager.get_transforms()
        self.class_names = self.model_manager.class_names
        
        # Performance optimization
        self.frame_count = 0
        self.process_every_n_frames = 2 # Process every 2nd frame to improve FPS
        self.last_predictions = []
        
        # Real-time stats tracking
        self.stats = InferenceStats()
        
        # Video state and detection tracking
        self.video_ended = False
        self.total_frames_processed = 0
        self.frames_with_faces = 0
        self.detection_counts = {}  # {name: count}
        self.unique_faces = []  # Store unique detected individuals 

    def __del__(self):
        if self.video:
            self.video.release()

    def set_model(self, backbone, classifier):
        print(f"Switching model to {backbone} + {classifier}")
        # Load model OUTSIDE the lock (this is the slow part)
        loaded_data = self.model_manager.load_model(backbone, classifier)
        
        # Get backbone attributes (input size)
        attributes = self.model_manager.get_attributes(backbone)
        new_input_size = attributes.get('input_size', 224)
        new_transform = self.model_manager.get_transforms(img_size=new_input_size)
        
        # Parse the loaded data
        if loaded_data[1] == 'pytorch':
            new_model = loaded_data[0]
            new_model_type = 'pytorch'
            new_backbone = None
            new_scaler = None
        else:
            new_model = loaded_data[0][0]  # Classifier
            new_scaler = loaded_data[0][1]  # Scaler
            new_backbone = loaded_data[0][2]  # Backbone
            new_model_type = 'sklearn'
        
        # Briefly lock to swap references (fast operation)
        with self.lock:
            self.current_model = new_model
            self.current_model_type = new_model_type
            self.current_backbone = new_backbone
            self.current_scaler = new_scaler
            self.transform = new_transform # Update transform!
        print(f"Model {backbone} + {classifier} loaded successfully. Input size: {new_input_size}x{new_input_size}")
             
    def set_video_source(self, source):
        """Switches video source to a file path or integer (webcam loop)."""
        print(f"Switching video source to {source}")
        with self.lock:
            if self.video:
                self.video.release()
                self.video = None
            new_video = cv2.VideoCapture(source)
            if not new_video.isOpened():
                print(f"Warning: Could not open video source {source}")
                return False
            self.video = new_video
            self.source_is_file = isinstance(source, str)
            # Reset detection stats for new video
            self.video_ended = False
            self.total_frames_processed = 0
            self.frames_with_faces = 0
            self.detection_counts = {}
            self.unique_faces = []
            self.current_video_path = source if isinstance(source, str) else None
        print(f"Video source set successfully.")
        return True
    
    def replay_video(self):
        """Replay the current video from the beginning."""
        with self.lock:
            if self.video and hasattr(self, 'source_is_file') and self.source_is_file:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.video_ended = False
                self.total_frames_processed = 0
                self.frames_with_faces = 0
                self.detection_counts = {}
                return True
        return False
    
    def get_detection_summary(self):
        """Returns a summary of detections for the video."""
        total = sum(self.detection_counts.values())
        summary = {
            'video_ended': self.video_ended,
            'total_frames': self.total_frames_processed,
            'frames_with_faces': self.frames_with_faces,
            'detection_percentage': (self.frames_with_faces / max(self.total_frames_processed, 1)) * 100,
            'detections': [
                {'name': name, 'count': count, 'percentage': (count / max(total, 1)) * 100}
                for name, count in sorted(self.detection_counts.items(), key=lambda x: -x[1])
            ],
            'total_detections': total
        }
        return summary

    def get_frame(self):
        start_time = time.time()
        
        with self.lock:
            if self.video is None:
                # Return placeholder when no video source
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "No Video Source", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(placeholder, "Please Upload a Video", (160, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                ret, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()
            
            # Check if video has already ended
            if self.video_ended:
                # Return "Video Ended" placeholder
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Video Analysis Complete", (130, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 245, 255), 2)
                cv2.putText(placeholder, f"Total Detections: {sum(self.detection_counts.values())}", (180, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                cv2.putText(placeholder, "Click 'Replay Video' to watch again", (120, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                ret, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()
            
            success, frame = self.video.read()
        
        if not success:
            # Video has ended - DON'T loop, just mark as ended
            with self.lock:
                if hasattr(self, 'source_is_file') and self.source_is_file:
                    self.video_ended = True
                    print(f"Video ended. Total detections: {sum(self.detection_counts.values())}")
            # Return the ended placeholder on next call
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Video Analysis Complete", (130, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 245, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', placeholder)
            return jpeg.tobytes()
        
        # Track frame count
        self.total_frames_processed += 1
        
        # Optimization: Resize frame for faster face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            rgb_small_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # If no model is loaded, just return frame with bounding boxes
        if self.current_model is None:
            for (x, y, w, h) in faces:
                # Scale back up
                x *= 4
                y *= 4
                w *= 4
                h *= 4
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, "No Model Loaded", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            latency_ms = (time.time() - start_time) * 1000
            self.stats.record_frame(latency_ms, [], len(faces))
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

        # Process faces
        predictions = []
        
        # Only process every N frames to save compute
        if self.frame_count % self.process_every_n_frames == 0:
            self.last_predictions = []
            
            for (x, y, w, h) in faces:
                # Scale coordinates back to original frame size
                x *= 4; y *= 4; w *= 4; h *= 4
                
                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w]
                rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                pil_face = Image.fromarray(rgb_face)
                
                try:
                    # Preprocess
                    input_tensor = self.transform(pil_face).unsqueeze(0).to(self.model_manager.device)
                    
                    # Inference
                    name = "Unknown"
                    conf = 0.0
                    
                    if self.current_model_type == 'pytorch':
                        with torch.no_grad():
                            start_time = cv2.getTickCount()
                            logits = self.current_model(input_tensor)
                            probs = torch.nn.functional.softmax(logits, dim=1)
                            conf, pred_idx = torch.max(probs, 1)
                            confidence = conf.item()
                            idx = pred_idx.item()
                            if idx < len(self.class_names):
                                name = self.class_names[idx]
                                
                    elif self.current_model_type == 'sklearn':
                        # Feature extraction
                        with torch.no_grad():
                            features = self.current_backbone(input_tensor)
                            features = features.cpu().numpy()
                            
                        # Scaling
                        if self.current_scaler:
                             features = self.current_scaler.transform(features)
                        
                        # Prediction
                        pred_idx = self.current_model.predict(features)[0]
                        # Try to get probability if available
                        if hasattr(self.current_model, 'predict_proba'):
                            probs = self.current_model.predict_proba(features)
                            confidence = np.max(probs)
                        else:
                            confidence = 1.0 # SVM/others might not have proba by default
                        
                        if pred_idx < len(self.class_names):
                            name = self.class_names[pred_idx]

                    self.last_predictions.append((x, y, w, h, name, confidence))
                    
                    # Track detection counts
                    if name != "Unknown" and name != "Error":
                        self.detection_counts[name] = self.detection_counts.get(name, 0) + 1
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
                    self.last_predictions.append((x, y, w, h, "Error", 0.0))
            
            # Track if this frame had faces
            if len(self.last_predictions) > 0:
                self.frames_with_faces += 1
        
        self.frame_count += 1
        
        for (x, y, w, h, name, conf) in self.last_predictions:
            color = (0, 255, 0) if conf > 0.6 else (0, 0, 255)
            label = f"{name} ({conf*100:.1f}%)"
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-35), (x+w, y), color, cv2.FILLED)
            cv2.putText(frame, label, (x+6, y-6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        # Record stats for real-time analytics
        latency_ms = (time.time() - start_time) * 1000
        confidences = [c for (_, _, _, _, _, c) in self.last_predictions]
        self.stats.record_frame(latency_ms, confidences, len(self.last_predictions))
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
