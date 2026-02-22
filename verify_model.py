import os
import torch
import joblib
import cv2
import numpy as np
import time
from model_loader import ModelManager
from models import FaceClassifier, get_backbone

# Mock config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model', 'part 2')

def verify_models():
    print(f"Checking models in {MODEL_DIR}")
    manager = ModelManager(MODEL_DIR)
    
    # 1. Load SVM Model with DEFAULT logic (which currently picks 'simple')
    print("\n--- Test 1: Load SVM with default logic ---")
    try:
        (svm, scaler, backbone), mode = manager.load_model('efficientnet_b3', 'svm_rbf')
        print(f"Loaded SVM: {type(svm)}")
        print(f"Loaded Scaler: {type(scaler)}")
        print(f"Loaded Backbone: {type(backbone)}")
        
        # Create dummy input (batch of 1)
        # 3 channels, 224x224
        # Random noise first
        dummy_input = torch.randn(1, 3, 224, 224).to(manager.device)
        
        # Run inference
        with torch.no_grad():
            features = backbone(dummy_input)
            if isinstance(features, tuple): features = features[0]
            
            features_np = features.cpu().numpy()
            if scaler:
                features_np = scaler.transform(features_np)
            
            # Predict
            pred = svm.predict(features_np)
            prob = svm.predict_proba(features_np) if hasattr(svm, 'predict_proba') else None
            
            print(f"Prediction (Random Input): {pred}")
            print(f"Prob: {prob}")
            
    except Exception as e:
        print(f"Test 1 Failed: {e}")

    # 2. Test 'Deep' Backbone weights instead of 'Simple'
    print("\n--- Test 2: Compare 'Simple' vs 'Deep' backbone weights ---")
    
    # helper to load specific backbone
    def load_specific_backbone(pt_filename):
        print(f"Loading backbone from {pt_filename}...")
        path = os.path.join(MODEL_DIR, pt_filename)
        if not os.path.exists(path):
            print(f"  -> File not found: {path}")
            return None
            
        bb, _ = get_backbone('efficientnet_b3', pretrained=True)
        try:
            checkpoint = torch.load(path, map_location=manager.device, weights_only=False)
        except Exception as e:
            print(f"  -> Failed to load {pt_filename}: {e}")
            return None
        
        # Extract backbone state
        state_dict = checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            
        backbone_state = {}
        for k, v in state_dict.items():
            if k.startswith('module.'): k = k[7:]
            if k.startswith('backbone.'):
                backbone_state[k.replace('backbone.', '')] = v
        
        if backbone_state:
            bb.load_state_dict(backbone_state, strict=False)
            print(f"  -> Successfully loaded {len(backbone_state)} keys.")
        else:
            print("  -> No backbone keys found!")
            # Try to load as full model if keys are missing?
            # Or inspect keys
            print(f"  -> Keys found: {list(state_dict.keys())[:5]}")
            
        bb.to(manager.device)
        bb.eval()
        return bb

    backbones = {
        'ImageNet (None)': get_backbone('efficientnet_b3', pretrained=True)[0].to(manager.device).eval(),
        'Simple': load_specific_backbone('efficientnet_b3_simple_run1.pt'),
        'Deep': load_specific_backbone('efficientnet_b3_deep_run1.pt'),
        'MLP': load_specific_backbone('efficientnet_b3_mlp_run1.pt')
    }
    
    # We need a REAL face image to check accuracy/confidence properly
    # Try to find a face in 'uploads' or generate a "face-like" blob?
    # Random noise won't give high confidence.
    # We will search for a jpg in the current directory or subdirs
    sample_img_path = None
    for root, dirs, files in os.walk(BASE_DIR):
        for f in files:
            if f.endswith('.jpg') or f.endswith('.png'):
                sample_img_path = os.path.join(root, f)
                break
        if sample_img_path: break
        
    if sample_img_path:
        with open('results.txt', 'w', encoding='utf-8') as f:
            f.write(f"USING_IMAGE: {os.path.basename(sample_img_path)}\n")
            try:
                # Load full image
                full_img = cv2.imread(sample_img_path)
                full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
                h_img, w_img = full_img.shape[:2]
                
                # Detect face to get a "Haar-like" crop
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(full_img, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    f.write(f"Detected Face: x={x}, y={y}, w={w}, h={h}\n")
                else:
                    f.write("No face detected by Haar. Using center crop.\n")
                    x, y = w_img//4, h_img//4
                    w, h = w_img//2, h_img//2

                # Load Neural Model (Simple) for testing
                pt_file = 'efficientnet_b3_simple_run1.pt'
                path = os.path.join(MODEL_DIR, pt_file)
                model = FaceClassifier(len(manager.class_names), 'efficientnet_b3', 'simple', pretrained=False)
                
                # Load weights carefully
                try:
                    ckpt = torch.load(path, map_location=manager.device, weights_only=False)
                    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
                    state = {k.replace('module.', ''): v for k, v in state.items()}
                    model.load_state_dict(state, strict=False)
                    model.to(manager.device).eval()
                    f.write("Loaded Simple Neural Model for Padding Test.\n")
                except Exception as e:
                    f.write(f"Failed to load neural model: {e}\n")
                    return

                transform = manager.get_transforms(224)
                f.write("\n--- PADDING TESTS ---\n")
                
                from PIL import Image
                paddings = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
                for pad in paddings:
                    # Apply margin
                    margin_w = int(w * pad)
                    margin_h = int(h * pad)
                    
                    x1 = max(0, x - margin_w)
                    y1 = max(0, y - margin_h)
                    x2 = min(w_img, x + w + margin_w)
                    y2 = min(h_img, y + h + margin_h)
                    
                    face_crop = full_img[y1:y2, x1:x2]
                    pil_crop = Image.fromarray(face_crop)
                    
                    input_tensor = transform(pil_crop).unsqueeze(0).to(manager.device)
                    
                    with torch.no_grad():
                        logits = model(input_tensor)
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                        max_prob = np.max(probs)
                        pred_idx = np.argmax(probs)
                        pred_name = manager.class_names[pred_idx] if pred_idx < len(manager.class_names) else "Unknown"
                        f.write(f"Pad={pad:.1f} (Size: {x2-x1}x{y2-y1}) -> Conf={max_prob:.4f} Class={pred_name}\n")

            except Exception as e:
                f.write(f"IMAGE_ERROR: {e}\n")
    else:
        with open('results.txt', 'w', encoding='utf-8') as f:
            f.write("NO_IMAGE_FOUND\n")

if __name__ == "__main__":
    verify_models()
