import json
from functools import lru_cache
from django.conf import settings
class SklearnRiskAdapter:
    @staticmethod
    @lru_cache(maxsize=1)
    def bundle():
        import joblib
        p=settings.BASE_DIR/'models/hybrid.joblib'
        return joblib.load(p) if p.exists() else None
    def infer(self,features):
        bundle=self.bundle()
        if bundle is None:return {'score':None,'classification':'Sin modelo','model_version':None}
        import numpy as np
        from ml.train import predict
        p=float(predict(bundle,np.array([[features[k] for k in bundle['features']]],dtype=float))[0])
        return {'score':p,'classification':'Clase experimental 1' if p>=.5 else 'Clase experimental 0','model_version':bundle['version']}
    def metrics(self):
        p=settings.BASE_DIR/'models/metrics.json'
        return json.loads(p.read_text()) if p.exists() else {'status':'Modelo no entrenado'}
