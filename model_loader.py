import os
import json
import joblib
import torch
import numpy as np
from torchvision import transforms
from models import FaceClassifier, get_backbone

# All known backbone prefixes for filename parsing
KNOWN_BACKBONES = [
    'convnext_tiny',
    'efficientnet_b3',
    'efficientnet_b0',
    'mobilenet_v3',
    'densenet121',
    'resnet101',
    'resnet50',
    'vit_b_16',
    'swin_t',
    'vgg16',
]

class ModelManager:
    def __init__(self, model_dirs):
        """
        Args:
            model_dirs: A single directory path (str) or a list of directory paths
                        where model files are stored.
        """
        if isinstance(model_dirs, str):
            model_dirs = [model_dirs]
        self.model_dirs = model_dirs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = self._load_class_names()
        self.scalers = {}  # Cache for scalers

    def _load_class_names(self):
        # Search all model dirs for class_names.json, use the first found
        for d in self.model_dirs:
            class_names_path = os.path.join(d, 'class_names.json')
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    return json.load(f)
        return []

    def _parse_backbone(self, filename):
        """Parse the backbone name from a model filename."""
        for backbone in KNOWN_BACKBONES:
            if filename.startswith(backbone + '_'):
                return backbone
        return None

    def get_available_models(self):
        """
        Scans all model directories and returns a structured dictionary of available models.
        Returns:
            dict: {
                'backbones': ['vgg16', 'resnet101', ...],
                'classifiers': {
                    'vgg16': ['simple', 'mlp', ...],
                    'resnet101': ['simple', 'svm_rbf', ...]
                }
            }
        """
        models = {'backbones': set(), 'classifiers': {}}

        for model_dir in self.model_dirs:
            if not os.path.isdir(model_dir):
                continue
            for filename in os.listdir(model_dir):
                if not (filename.endswith('.pt') or filename.endswith('.joblib')):
                    continue
                if 'scaler' in filename:
                    continue

                backbone = self._parse_backbone(filename)
                if backbone:
                    models['backbones'].add(backbone)
                    if backbone not in models['classifiers']:
                        models['classifiers'][backbone] = set()

                    # Extract classifier name: remove backbone prefix and _run1.ext suffix
                    classifier_part = filename[len(backbone) + 1:]  # +1 for the underscore
                    classifier = classifier_part.rsplit('_run1', 1)[0]
                    models['classifiers'][backbone].add(classifier)

        models['backbones'] = sorted(list(models['backbones']))
        for k in models['classifiers']:
            models['classifiers'][k] = sorted(list(models['classifiers'][k]))

        return models

    def _find_file(self, filename):
        """Search all model directories for a file, return full path or None."""
        for d in self.model_dirs:
            path = os.path.join(d, filename)
            if os.path.exists(path):
                return path
        return None

    def get_attributes(self, backbone_name):
        """Returns attributes like input size for a specific backbone."""
        # All models use IMG_SIZE = 224
        return {'input_size': 224}

    def get_transforms(self, img_size=224):
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, backbone_name, classifier_name):
        """
        Loads a specific model configuration.
        Returns:
            model: The loaded model (PyTorch nn.Module or Sklearn estimator)
            model_type: 'pytorch' or 'sklearn'
            preprocess: Transform function/pipeline
        """
        # Check if it's a PyTorch model
        pt_filename = f"{backbone_name}_{classifier_name}_run1.pt"
        pt_path = self._find_file(pt_filename)

        if pt_path:
            return self._load_pytorch_model(pt_path, backbone_name, classifier_name)

        # Check if it's a Scikit-learn model
        joblib_filename = f"{backbone_name}_{classifier_name}_run1.joblib"
        joblib_path = self._find_file(joblib_filename)

        if joblib_path:
            return self._load_sklearn_model(joblib_path, backbone_name)

        raise FileNotFoundError(f"Model not found for {backbone_name} + {classifier_name}")

    def _load_pytorch_model(self, model_path, backbone_name, head_name):
        print(f"Loading PyTorch model from {model_path}...")
        model = FaceClassifier(
            num_classes=len(self.class_names),
            backbone_name=backbone_name,
            head_name=head_name,
            pretrained=False  # Loading trained weights, initial weights don't matter
        )

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict) and all(k.startswith('backbone.') or k.startswith('head.') for k in list(checkpoint.keys())[:5]):
                model.load_state_dict(checkpoint)
            else:
                try:
                    model.load_state_dict(checkpoint)
                except:
                    print("Could not load state dict directly. Assuming entire model was saved.")
                    model = checkpoint

            model.to(self.device)
            model.eval()
            return model, 'pytorch'
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    def _load_sklearn_model(self, model_path, backbone_name):
        print(f"Loading Scikit-learn model from {model_path}...")
        loaded_obj = joblib.load(model_path)

        # Handle case where model is wrapped in a dictionary
        if isinstance(loaded_obj, dict):
            if 'classifier' in loaded_obj:
                classifier = loaded_obj['classifier']
                print(f"  -> Extracted classifier from dict: {type(classifier)}")
            elif 'model' in loaded_obj:
                classifier = loaded_obj['model']
                print(f"  -> Extracted model from dict: {type(classifier)}")
            else:
                raise ValueError(f"Joblib file contains dict but no 'classifier' or 'model' key. Keys: {list(loaded_obj.keys())}")
        else:
            classifier = loaded_obj

        # Load scaler
        scaler_filename = f"{backbone_name}_scaler.joblib"
        if backbone_name not in self.scalers:
            scaler_path = self._find_file(scaler_filename)
            if scaler_path:
                print(f"Loading Scaler from {scaler_path}...")
                scaler_obj = joblib.load(scaler_path)
                if isinstance(scaler_obj, dict) and 'scaler' in scaler_obj:
                    self.scalers[backbone_name] = scaler_obj['scaler']
                else:
                    self.scalers[backbone_name] = scaler_obj
            else:
                print(f"Warning: Scaler not found for {backbone_name}. Inference might be incorrect.")
                self.scalers[backbone_name] = None

        scaler = self.scalers[backbone_name]

        # Load Backbone Feature Extractor
        print(f"Loading Backbone {backbone_name} for feature extraction...")
        backbone, out_features = get_backbone(backbone_name, pretrained=True)

        # Try to load fine-tuned backbone weights if available
        found_weights = False
        potential_weights = [
            f"{backbone_name}_simple_run1.pt",
            f"{backbone_name}_mlp_run1.pt",
            f"{backbone_name}_deep_run1.pt"
        ]

        for pt_file in potential_weights:
            pt_path = self._find_file(pt_file)
            if pt_path:
                print(f"Found potential fine-tuned weights: {pt_file}. Loading backbone from it...")
                try:
                    checkpoint = torch.load(pt_path, map_location=self.device, weights_only=False)

                    backbone_state = {}
                    state_dict = checkpoint
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']

                    for k, v in state_dict.items():
                        if k.startswith('module.'):
                            k = k[7:]
                        if k.startswith('backbone.'):
                            backbone_state[k.replace('backbone.', '')] = v

                    if backbone_state:
                        msg = backbone.load_state_dict(backbone_state, strict=False)
                        print(f"Loaded fine-tuned backbone weights from {pt_file}. Missing keys: {len(msg.missing_keys)}")
                        found_weights = True
                        break
                except Exception as e:
                    print(f"Failed to load weights from {pt_file}: {e}")

        if not found_weights:
            print(f"Warning: No fine-tuned weights found for {backbone_name}. Using ImageNet weights.")

        backbone.to(self.device)
        backbone.eval()

        return (classifier, scaler, backbone), 'sklearn'
