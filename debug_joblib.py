import joblib
import os

path = r'c:/Users/yassi/Documents/s3/Deep Learning/projet face recognition/model/part 2'
files = [f for f in os.listdir(path) if f.endswith('.joblib') and 'scaler' not in f]

for f in files[:3]:
    print(f'\n=== File: {f} ===')
    m = joblib.load(os.path.join(path, f))
    print(f'Type: {type(m)}')
    if isinstance(m, dict):
        print(f'Keys: {list(m.keys())}')
        for k, v in list(m.items())[:2]:
            print(f'  {k}: {type(v)}')
    elif hasattr(m, 'predict'):
        print('Has predict method: True')
    else:
        print(f'Object info: {dir(m)[:5]}...')
