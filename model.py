"""EfficientNet-B0, DenseNet121 and checkpoint-backed wildlife prediction."""
import hashlib
import re
from pathlib import Path

import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, densenet121, DenseNet121_Weights

from dataset import build_transform


class WildlifeEfficientNetB0(nn.Module):
    architecture = 'efficientnet_b0'
    fine_tuning_first_block = 6

    def __init__(self, num_classes, pretrained=False, dropout_rate=0.2, weights_path=None):
        super().__init__()
        if num_classes < 2:
            raise ValueError('At least two known classes are required')
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained and not weights_path else None
        self.model = efficientnet_b0(weights=weights)
        if weights_path:
            path = Path(weights_path)
            if not hashlib.sha256(path.read_bytes()).hexdigest().startswith('7f5810bc'):
                raise ValueError('Expected official EfficientNet-B0 IMAGENET1K_V1 weights')
            self.model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(features, num_classes))

    def extract_features(self, images):
        return self.model.avgpool(self.model.features(images)).flatten(1)

    def forward(self, images):
        return self.model.classifier(self.extract_features(images))

    def set_trainable_blocks(self, first_block=None):
        for parameter in self.model.features.parameters():
            parameter.requires_grad_(False)
        if first_block is not None:
            if first_block < 0 or first_block >= len(self.model.features):
                raise ValueError('Invalid first trainable block')
            for block in self.model.features[first_block:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)

    def training_mode(self):
        self.train()
        for block in self.model.features:
            if not any(p.requires_grad for p in block.parameters()):
                block.eval()
        # Preserve pretrained BatchNorm statistics with small transfer-learning batches.
        for module in self.model.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class WildlifeDenseNet121(WildlifeEfficientNetB0):
    architecture = 'densenet121'
    fine_tuning_first_block = 10  # denseblock4 and norm5

    def __init__(self, num_classes, pretrained=False, dropout_rate=0.2, weights_path=None):
        nn.Module.__init__(self)
        if num_classes < 2:
            raise ValueError('At least two known classes are required')
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained and not weights_path else None
        self.model = densenet121(weights=weights)
        if weights_path:
            path = Path(weights_path)
            if not hashlib.sha256(path.read_bytes()).hexdigest().startswith('a639ec97'):
                raise ValueError('Expected official DenseNet121 IMAGENET1K_V1 weights')
            state = torch.load(path, map_location='cpu', weights_only=True)
            # Official weights retain old dotted layer names; Torchvision performs
            # this same conversion when downloading weights through its builder.
            pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.([12]\.(?:weight|bias|running_mean|running_var))$')
            for key in list(state):
                match = pattern.match(key)
                if match:
                    state[match.group(1) + match.group(2)] = state.pop(key)
            self.model.load_state_dict(state)
        features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(features, num_classes))

    def extract_features(self, images):
        features = nn.functional.relu(self.model.features(images), inplace=False)
        return nn.functional.adaptive_avg_pool2d(features, (1, 1)).flatten(1)


def build_model(architecture, num_classes, **kwargs):
    builders = {'efficientnet_b0': WildlifeEfficientNetB0, 'densenet121': WildlifeDenseNet121}
    if architecture not in builders:
        raise ValueError(f'Unsupported architecture: {architecture}')
    return builders[architecture](num_classes, **kwargs)


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    mapping = checkpoint['class_to_idx']
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError('Checkpoint class mapping is not contiguous')
    architecture = checkpoint.get('architecture', checkpoint.get('config', {}).get('architecture', 'efficientnet_b0'))
    model = build_model(architecture, len(mapping))
    model.load_state_dict(checkpoint['model_state'])
    return model.to(device).eval(), checkpoint


class OpenSetWildlifePredictor:
    """Single-image prediction with a threshold selected from calibration data."""
    def __init__(self, trained_model, confidence_threshold, class_names, temperature=1.0):
        if not 0 <= confidence_threshold <= 1.000001 or temperature <= 0:
            raise ValueError('Invalid calibration parameters')
        if len(class_names) != trained_model.model.classifier[1].out_features:
            raise ValueError('Class names do not match the classifier')
        self.model = trained_model.eval()
        self.threshold = confidence_threshold
        self.class_names = class_names
        self.temperature = temperature
        self.transform = build_transform(training=False)

    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        model, checkpoint = load_checkpoint(path, device)
        calibration = checkpoint.get('calibration')
        if not calibration:
            raise ValueError('Checkpoint has not been calibrated for unknown rejection')
        names = sorted(checkpoint['class_to_idx'], key=checkpoint['class_to_idx'].get)
        return cls(model, calibration['threshold'], names, calibration['temperature'])

    @torch.inference_mode()
    def predict(self, image):
        tensor = image if isinstance(image, torch.Tensor) else self.transform(image)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tuple(tensor.shape) != (1, 3, 224, 224):
            raise ValueError('Expected one RGB image transformed to 224 x 224')
        tensor = tensor.to(next(self.model.parameters()).device)
        probabilities = (self.model(tensor) / self.temperature).softmax(1)
        confidence, index = probabilities.max(1)
        confidence, index = float(confidence.item()), int(index.item())
        known = confidence >= self.threshold
        return {'category': self.class_names[index] if known else 'Unknown',
                'confidence': confidence, 'is_known': known,
                'closest_known_category': self.class_names[index]}
