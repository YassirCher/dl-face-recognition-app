import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model', 'part 2')
filename = 'efficientnet_b3_simple_run1.pt'
path = os.path.join(MODEL_DIR, filename)

print(f"Inspecting {path}...")
try:
    with open(path, 'rb') as f:
        header = f.read(200)
        print(f"First 200 bytes: {header}")
        try:
            print(f"Decoded: {header.decode('utf-8')}")
        except:
            print("Cannot decode as utf-8 (binary)")
except Exception as e:
    print(f"Error: {e}")
