import torch
import os

model_path = r"c:/Users/yassi/Documents/s3/Deep Learning/projet face recognition/model/part 2/efficientnet_b3_cosface_run1.pt"
print(f"Torch version: {torch.__version__}")

try:
    print("\nAttempt 1: Loading with default...")
    checkpoint = torch.load(model_path)
    print("Success with default!")
except Exception as e:
    print(f"Failed with default: {e}")

try:
    print("\nAttempt 2: Loading with weights_only=False...")
    checkpoint = torch.load(model_path, weights_only=False)
    print("Success with weights_only=False!")
except Exception as e:
    print(f"Failed with weights_only=False: {e}")
    
try:
    print("\nAttempt 3: Loading with weights_only=True...")
    checkpoint = torch.load(model_path, weights_only=True)
    print("Success with weights_only=True!")
except Exception as e:
    print(f"Failed with weights_only=True: {e}")
