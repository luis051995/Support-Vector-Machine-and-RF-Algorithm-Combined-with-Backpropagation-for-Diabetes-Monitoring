"""Entrenamiento reproducible. Nunca entrena con expedientes de la aplicación."""
import argparse, hashlib, json, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score, brier_score_loss
from sklearn.exceptions import ConvergenceWarning
import sklearn
ROOT=Path(__file__).resolve().parents[1]
FEATURES=['Edad','Sexo','Peso','Altura','Antecedentes','Acetona']

def clean(path):
    df=pd.read_csv(path, encoding='utf-8-sig')
    missing=set(FEATURES+['Salida'])-set(df)
    if missing: raise ValueError(f'Columnas faltantes: {missing}')
    raw=len(df)
    for col in ['Sexo','Antecedentes']:
        mapping={'mujer':0,'hombre':1,'no':0,'yes':1,'sí':1,'si':1,'0':0,'1':1}
        df[col]=df[col].astype(str).str.strip().str.lower().map(mapping)
    for col in FEATURES+['Salida']: df[col]=pd.to_numeric(df[col],errors='coerce')
    meters=df.Altura.between(.8,2.5); df.loc[meters,'Altura']*=100
    valid=(df.Edad.between(18,100)&df.Peso.between(30,300)&df.Altura.between(100,230)&df.Acetona.between(0,100)&df.Sexo.isin([0,1])&df.Antecedentes.isin([0,1])&df.Salida.isin([0,1]))
    rejected=df.loc[~valid].copy(); df=df.loc[valid].copy()
    before=len(df); df=df.drop_duplicates(subset=FEATURES+['Salida'])
    # Sin ID no se puede demostrar independencia por persona. Agrupación demográfica conservadora.
    if 'patient_id' in df and df.patient_id.notna().all(): groups=df.patient_id.astype(str)
    else: groups=df[['Edad','Sexo','Peso','Altura']].astype(str).agg('|'.join,axis=1)
    conflicts=int(df.groupby(FEATURES).Salida.nunique().gt(1).sum())
    audit={'original_rows':raw,'height_rows_converted':int(meters.sum()),'rejected_rows':len(rejected),'duplicates_removed':before-len(df),'usable_rows':len(df),'conflicting_feature_groups':conflicts,'class_counts':{str(k):int(v) for k,v in df.Salida.value_counts().items()},'grouping':'patient_id' if 'patient_id' in df else 'surrogate_demographics','limitation':'CSV sin identidad ni procedencia clínica verificadas. Grupos demográficos aproximados; se requiere validación externa por paciente.'}
    return df.reset_index(drop=True),np.array(groups),audit,rejected

def fit_stack(X,y,groups):
    cv=list(StratifiedGroupKFold(4,shuffle=True,random_state=23).split(X,y,groups))
    base=[make_pipeline(StandardScaler(),SVC(C=1,gamma='scale',probability=True,class_weight='balanced',random_state=23)),RandomForestClassifier(n_estimators=160,max_depth=5,min_samples_leaf=3,class_weight='balanced',random_state=23,n_jobs=1)]
    svm_search=GridSearchCV(base[0],{'svc__C':[.1,1,10],'svc__gamma':['scale',.01]},cv=cv,scoring='roc_auc',n_jobs=1).fit(X,y)
    base[0]=clone(svm_search.best_estimator_)
    oof=np.zeros((len(y),2))
    for j,est in enumerate(base):
        for tr,va in cv:
            oof[va,j]=clone(est).fit(X[tr],y[tr]).predict_proba(X[va])[:,1]
    meta=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(8,),solver='lbfgs',alpha=1.,max_iter=800,random_state=23))
    meta.fit(oof,y)
    for est in base: est.fit(X,y)
    return base,meta,svm_search.best_params_

def predict(bundle,X):
    return bundle['meta'].predict_proba(np.column_stack([e.predict_proba(X)[:,1] for e in bundle['base']]))[:,1]

def metrics(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1]).ravel()
    return {'accuracy':accuracy_score(y,p>=.5),'roc_auc':roc_auc_score(y,p),'f1':f1_score(y,p>=.5,zero_division=0),'sensitivity':float(tp/max(1,tp+fn)),'specificity':float(tn/max(1,tn+fp)),'brier':brier_score_loss(y,p),'confusion_matrix':[[int(tn),int(fp)],[int(fn),int(tp)]]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--csv',default=str(ROOT/'data/diabetes_original.csv'));a=ap.parse_args()
    df,groups,audit,rejected=clean(a.csv); X=df[FEATURES].to_numpy(float);y=df.Salida.to_numpy(int)
    tr,te=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(X,y,groups))
    assert not set(groups[tr])&set(groups[te])
    base,meta,params=fit_stack(X[tr],y[tr],groups[tr])
    digest=hashlib.sha256(Path(a.csv).read_bytes()).hexdigest()
    bundle={'base':base,'meta':meta,'features':FEATURES,'version':'research-v2-'+digest[:10],'sklearn':sklearn.__version__,'threshold':.5,'validated_clinically':False}
    report={'version':bundle['version'],'data_audit':audit,'train_rows':len(tr),'test_rows':len(te),'split':'StratifiedGroupKFold; holdout no usado en ajuste; grupos disjuntos','best_svm_params':params,'hybrid':metrics(y[te],predict(bundle,X[te])),'svm_reference':metrics(y[te],base[0].predict_proba(X[te])[:,1]),'rf_reference':metrics(y[te],base[1].predict_proba(X[te])[:,1]),'note':'Comparación interna sobre el mismo holdout. No demuestra superioridad clínica ni reproduce la red TensorFlow original. Probabilidades sin calibración clínica.'}
    out=ROOT/'models';out.mkdir(exist_ok=True)
    joblib.dump(bundle,out/'hybrid.joblib');(out/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    rejected.to_csv(ROOT/'data/revision_excluidos.csv',index=False)
    print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
