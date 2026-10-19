"""The same checkpoint-backed prediction logic used by the verified local demo."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from model import OpenSetWildlifePredictor
from reid_model import DualHeadPredictor
from retrieval_refinement import GalleryMatcher, normalize
ROOT=Path(__file__).resolve().parent
def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Models:
    def __init__(self, model_dir, image_root, refinement_dir=None):
        self.image_root = Path(image_root).resolve()
        self.classifier = OpenSetWildlifePredictor.from_checkpoint(model_dir/'efficientnet_b0.pt')
        self.dual = DualHeadPredictor(model_dir/'dual_head_efficientnet.pt', model_dir/'reference_gallery.pt')
        gallery = torch.load(model_dir/'reference_gallery.pt', map_location='cpu', weights_only=True)
        self.gallery_ids = gallery['sample_ids']
        self.refinement = None
        self.matcher = None
        if refinement_dir is not None:
            refinement_dir=Path(refinement_dir)
            config=json.loads((refinement_dir/'deployment.json').read_text())
            if config.get('version') != 1 or not config.get('release_passed'):
                raise ValueError('Retrieval refinement did not pass its release checks')
            if config['checkpoint_sha256'] != fingerprint(model_dir/'dual_head_efficientnet.pt'):
                raise ValueError('Refinement belongs to a different model')
            if config['gallery_sha256'] != fingerprint(refinement_dir/'refined_gallery.pt'):
                raise ValueError('Refinement gallery fingerprint does not match')
            refined=torch.load(refinement_dir/'refined_gallery.pt',map_location='cpu',weights_only=True)
            if (refined['sample_ids'] != self.gallery_ids or refined['identities'] != self.dual.identities
                    or not torch.equal(torch.as_tensor(refined['species_ids']),self.dual.species)):
                raise ValueError('Refined gallery does not match the original reference IDs')
            vectors=refined['embeddings'].numpy()
            if vectors.shape != tuple(self.dual.gallery.shape) or not np.isfinite(vectors).all():
                raise ValueError('Invalid refined reference features')
            self.matcher=GalleryMatcher(vectors,refined['identities'],refined['species_ids'],
                                        center=config['center'],density=config['density'])
            self.refinement=config
        if len(self.gallery_ids) != len(self.dual.gallery) or len(set(self.gallery_ids)) != len(self.gallery_ids):
            raise ValueError('Invalid gallery image IDs')
        frame = pd.read_csv(ROOT/'data/prepared/open_set/test.csv.gz', keep_default_na=False).set_index('sample_id')
        self.gallery_rows = {}
        for i, uid in enumerate(self.gallery_ids):
            row = frame.loc[uid]
            if row.identity != self.dual.identities[i] or int(row.class_id) != int(self.dual.species[i]):
                raise ValueError('Gallery labels do not match the shared manifest')
            self.gallery_rows[uid] = row.to_dict()
        self.health = {
            'status':'ready', 'model':'EfficientNet-B0', 'categories':self.classifier.class_names,
            'classification_threshold':self.classifier.threshold,
            'dual_species_threshold':self.dual.calibration['species_threshold'],
            'identity_threshold':self.refinement['threshold'] if self.refinement else self.dual.calibration['identity_threshold'],
            'identity_matching':'refined' if self.refinement else 'original_cosine',
            'refinement_sha256':fingerprint(Path(refinement_dir)/'deployment.json') if self.refinement else None,
            'gallery_images':len(self.gallery_ids), 'gallery_individuals':len(set(self.dual.identities)),
            'checkpoint_sha256':{name:fingerprint(model_dir/name) for name in
                ['efficientnet_b0.pt','dual_head_efficientnet.pt','reference_gallery.pt']}}

    @torch.inference_mode()
    def predict(self, image, mode):
        if mode == 'classification':
            p = self.classifier
            x = p.transform(image).unsqueeze(0)
            probabilities = (p.model(x)/p.temperature).softmax(1)[0]
            names, threshold = p.class_names, p.threshold
        else:
            p = self.dual
            x = p.transform(image).unsqueeze(0)
            logits, embedding = p.model(x)
            if self.refinement and self.refinement['flip_average']:
                _, flipped = p.model(x.flip(-1))
                embedding = torch.nn.functional.normalize(embedding+flipped,dim=1)
            probabilities = (logits/p.calibration['temperature']).softmax(1)[0]
            names, threshold = p.names, p.calibration['species_threshold']
        confidence, label = probabilities.max(0)
        confidence, label = float(confidence), int(label)
        known = confidence >= threshold
        scores, indices = probabilities.topk(min(5, len(names)))
        result = {'mode':mode, 'category':names[label] if known else 'Unknown',
                  'closest_known_category':names[label], 'confidence':confidence,
                  'is_known':known, 'species_threshold':threshold,
                  'top_species':[{'category':names[int(i)],'confidence':float(s)} for s,i in zip(scores,indices)]}
        if mode == 'identification':
            result.update(individual_id=None, is_known_individual=False, individual_similarity=None,
                          identity_threshold=self.health['identity_threshold'], matches=[],
                          identity_score=None,identity_score_kind='adjusted_match' if self.refinement else 'cosine')
            candidates = torch.nonzero(p.species.eq(label)).flatten()
            if known and candidates.numel():
                if self.matcher is not None:
                    vector=embedding[0].numpy()
                    decision=self.matcher.decision(vector,label,margin_weight=self.refinement['margin_weight'])
                    result['identity_score']=decision['decision_score']
                    result['identity_margin']=decision['margin']
                    for rank,(index,score) in enumerate(zip(decision['indices'],decision['scores']),1):
                        uid=self.gallery_ids[index]
                        similarity=float(self.matcher.embeddings[index] @ normalize(vector))
                        result['matches'].append({'rank':rank,'identity':p.identities[index],
                            'similarity':similarity,'match_score':float(score),'sample_id':uid,
                            'image_url':f'/gallery/{uid}'})
                    if result['matches']:
                        result['individual_similarity']=result['matches'][0]['similarity']
                        if result['identity_score']>=self.refinement['threshold']:
                            result['individual_id']=result['matches'][0]['identity']
                            result['is_known_individual']=True
                    return result
                similarities = p.gallery[candidates] @ embedding[0]
                values, ranks = similarities.topk(min(5, len(candidates)))
                result['individual_similarity'] = float(values[0])
                result['identity_score'] = float(values[0])
                for rank, (score, local) in enumerate(zip(values,ranks),1):
                    index = int(candidates[int(local)])
                    uid = self.gallery_ids[index]
                    result['matches'].append({'rank':rank,'identity':p.identities[index],
                        'similarity':float(score),'sample_id':uid, 'image_url':f'/gallery/{uid}'})
                if float(values[0]) >= p.calibration['identity_threshold']:
                    result['individual_id'] = result['matches'][0]['identity']
                    result['is_known_individual'] = True
        return result
