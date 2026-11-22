import runpy, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; plt.show=lambda *a,**k:None
import numpy as np
from pathlib import Path
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
# Reuse plot_figure2.py's exact out-of-fold train/test split (it must sit in the same folder as this script)
g=runpy.run_path(str(Path(__file__).resolve().parent / 'plot_figure2.py'))
train,test,ytr,yte=g['train'],g['test'],g['y_train'],g['y_test']
weather=['FH','FX','T','RH','DR','VV','is_gale','is_freezing','S','Y','O','M']
def prep(num,flags,cat):
    t=[('num',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),num)]
    if flags: t.append(('f',SimpleImputer(strategy='constant',fill_value=0),flags))
    t.append(('c',Pipeline([('i',SimpleImputer(strategy='constant',fill_value='missing')),('o',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat))
    return ColumnTransformer(t)
def run(clf,num,flags,cat=['country','type']):
    p=Pipeline([('p',prep(num,flags,cat)),('c',clone(clf))]).fit(train[num+flags+cat],ytr)
    return roc_auc_score(yte,p.predict_proba(test[num+flags+cat])[:,1])
rf=RandomForestClassifier(n_estimators=200,max_depth=12,min_samples_leaf=10,random_state=42,n_jobs=-1)
num=g['num_cols']; flags=g['flag_cols']
print('RF A full        %.4f'%run(rf,num,flags))
print('RF A no weather  %.4f'%run(rf,[c for c in num if c not in weather],[]))
print('ExtraTrees A     %.4f'%run(ExtraTreesClassifier(n_estimators=300,max_depth=12,min_samples_leaf=10,random_state=42,n_jobs=-1),num,flags))
print('HistGB A         %.4f'%run(HistGradientBoostingClassifier(max_iter=200,max_depth=6,learning_rate=0.05,min_samples_leaf=30,l2_regularization=1.0,random_state=42),num,flags))
