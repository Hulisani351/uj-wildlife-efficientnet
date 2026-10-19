"""Lisa's dual-head EfficientNet architecture (commit e285ed6), training and inference."""
import torch
from torch import nn
from dataset import build_transform
import torchvision.models as models
import torch.nn.functional as F
from model import WildlifeEfficientNetB0 as ImageNetBackbone


class LisaEfficientNet(nn.Module):
    """
    EfficientNet-B0 customized for:
    1. Species Classification (Class Logits)
    2. Individual Identification Re-ID (L2-Normalized Embedding Vector)
    """
    def __init__(self, num_classes=11, embedding_dim=512, pretrained=True, dropout_rate=0.2):
        super(LisaEfficientNet, self).__init__()

        # Load weights configuration: default to ImageNet pretrained weights if requested
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None

        # Initialize the base EfficientNet-B0 backbone architecture
        base_model = models.efficientnet_b0(weights=weights)

        # Extract feature extractor backbone (excludes original classifier head)
        self.backbone = base_model.features
        self.avgpool = base_model.avgpool

        # Feature dimension out of EfficientNet-B0 backbone is 1280
        in_features = 1280

        # 1. Feature Embedding Head for Individual Re-Identification (ID)
        self.embedding_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        # 2. Classifier Head for Species/Category Classification
        self.species_classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        # Extract feature maps from backbone: [Batch, 1280, 7, 7]
        features = self.backbone(x)

        # Global Average Pooling: [Batch, 1280, 1, 1] -> Flatten: [Batch, 1280]
        pooled_features = torch.flatten(self.avgpool(features), 1)

        # Compute feature vector for Individual Identification
        embeddings = self.embedding_head(pooled_features)

        # Normalize embeddings to unit length (L2 norm) for cosine distance metric matching
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute species logits from identity embeddings
        species_logits = self.species_classifier(embeddings)

        # Return both species classification logits and individual feature embeddings
        return species_logits, norm_embeddings



class DualHeadWildlife(LisaEfficientNet):
    def __init__(self, num_classes, weights_path=None, embedding_dim=512):
        super().__init__(num_classes=num_classes, embedding_dim=embedding_dim, pretrained=False)
        if weights_path is not None:
            source = ImageNetBackbone(num_classes, weights_path=weights_path)
            self.backbone.load_state_dict(source.model.features.state_dict())

    def extract_features(self, images):
        return self.avgpool(self.backbone(images)).flatten(1)

    def heads(self, features):
        embedding = self.embedding_head(features)
        return self.species_classifier(embedding), nn.functional.normalize(embedding, dim=1)

    def forward(self, images):
        return self.heads(self.extract_features(images))

    def set_trainable(self, first_block=None):
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if first_block is not None:
            for p in self.backbone[first_block:].parameters():
                p.requires_grad_(True)

    def training_mode(self):
        self.train()
        for block in self.backbone:
            if not any(p.requires_grad for p in block.parameters()):
                block.eval()
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def identity_loss(embeddings, identities, species, margin=.2):
    """Batch-hard cosine triplets; negatives must be DIFFERENT animals of the SAME species."""
    distance = (1 - embeddings @ embeddings.T).clamp_min(0)
    same = identities[:, None].eq(identities[None, :])
    positive = same & ~torch.eye(len(identities), dtype=torch.bool, device=identities.device)
    negative = ~same & species[:, None].eq(species[None, :])
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        return embeddings.sum() * 0
    hardest_positive = distance.masked_fill(~positive, -torch.inf).max(1).values
    hardest_negative = distance.masked_fill(~negative, torch.inf).min(1).values
    return (hardest_positive[valid] - hardest_negative[valid] + margin).clamp_min(0).mean()


class DualHeadPredictor:
    """Predict using a calibrated checkpoint and genuine labeled gallery embeddings."""
    def __init__(self, checkpoint_path, gallery_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        self.calibration = checkpoint['calibration']
        mapping = checkpoint['class_to_idx']
        self.names = sorted(mapping, key=mapping.get)
        self.model = DualHeadWildlife(len(mapping), embedding_dim=checkpoint['embedding_dim'])
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        gallery = torch.load(gallery_path, map_location='cpu', weights_only=True)
        self.gallery = nn.functional.normalize(gallery['embeddings'], dim=1)
        self.identities = gallery['identities']
        self.species = torch.tensor(gallery['species_ids'])
        if self.gallery.ndim != 2 or self.gallery.shape[1] != checkpoint['embedding_dim'] or len(self.gallery) != len(self.identities) or len(self.species) != len(self.identities):
            raise ValueError('Gallery dimensions do not match the checkpoint')
        self.transform = build_transform(False)

    @torch.no_grad()
    def predict(self, image):
        x = image if isinstance(image, torch.Tensor) else self.transform(image)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if tuple(x.shape) != (1,3,224,224):
            raise ValueError('Expected one RGB image at 224 x 224')
        logits, embedding = self.model(x.cpu())
        probabilities = (logits/self.calibration['temperature']).softmax(1)
        confidence, label = probabilities.max(1)
        label, confidence = int(label.item()), float(confidence.item())
        known = confidence >= self.calibration['species_threshold']
        result = {'category': self.names[label] if known else 'Unknown', 'confidence': confidence,
                  'is_known': known, 'closest_known_category': self.names[label],
                  'individual_id': None, 'is_known_individual': False, 'individual_similarity': None}
        candidates = self.species.eq(label)
        if known and candidates.any():
            indices = torch.nonzero(candidates).flatten()
            similarities = self.gallery[indices] @ embedding[0]
            best = int(similarities.argmax())
            score = float(similarities[best])
            result['individual_similarity'] = score
            if score >= self.calibration['identity_threshold']:
                result['individual_id'] = self.identities[int(indices[best])]
                result['is_known_individual'] = True
        return result
