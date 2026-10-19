"""Gallery-only centering and cross-identity density correction for retrieval."""
import numpy as np


def normalize(x):
    x=np.asarray(x,dtype=np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


class GalleryMatcher:
    def __init__(self,embeddings,identities,species,center=0.,density=0.):
        self.embeddings=normalize(embeddings)
        self.identities=np.asarray(identities)
        self.species=np.asarray(species)
        self.center=float(center)
        self.density=float(density)
        self.groups={}
        for label in np.unique(self.species):
            indices=np.flatnonzero(self.species==label)
            g=self.embeddings[indices]
            mean=g.mean(0,keepdims=True)
            transformed=normalize(g-self.center*mean)
            bias=np.zeros(len(g),dtype=np.float32)
            if self.density:
                pairs=transformed@transformed.T
                ids=self.identities[indices]
                for j in range(len(g)):
                    other=pairs[j,ids!=ids[j]]
                    if len(other):
                        bias[j]=np.sort(other)[-min(3,len(other)):].mean()
            self.groups[int(label)]=(indices,mean,transformed,bias)

    def score(self,queries,predicted_species):
        q=normalize(queries)
        predicted_species=np.asarray(predicted_species)
        scores=np.full((len(q),len(self.embeddings)),-1e9,dtype=np.float32)
        for label,(gi,mean,g,bias) in self.groups.items():
            qi=np.flatnonzero(predicted_species==label)
            if not len(qi):
                continue
            adjusted=normalize(q[qi]-self.center*mean)
            scores[np.ix_(qi,gi)]=adjusted@g.T-self.density*bias[None,:]
        return scores

    def decision(self,query,predicted_species,margin_weight=0.,limit=5):
        scores=self.score(np.asarray(query).reshape(1,-1),[predicted_species])[0]
        ranking=np.argsort(-scores,kind='stable')
        ranking=ranking[scores[ranking]>-1e8]
        if not len(ranking):
            return {'indices':[],'scores':[],'decision_score':-1e9,'margin':0.}
        best=int(ranking[0])
        others=ranking[self.identities[ranking]!=self.identities[best]]
        margin=float(scores[best]-scores[others[0]]) if len(others) else 0.
        return {'indices':ranking[:limit].tolist(),'scores':scores[ranking[:limit]].tolist(),
                'decision_score':float(scores[best]+margin_weight*margin),'margin':margin}
