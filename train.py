import os, time, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

df=pd.read_csv('ai4i2020.csv').rename(columns={'Type':'type','Air temperature [K]':'air_temp','Process temperature [K]':'process_temp','Rotational speed [rpm]':'rpm','Torque [Nm]':'torque','Tool wear [min]':'tool_wear','Machine failure':'failure'})
df=df.drop(columns=['UDI','Product ID','TWF','HDF','PWF','OSF','RNF'])
NUM=['air_temp','process_temp','rpm','torque','tool_wear']; CAT=['type']
X,y=df[NUM+CAT],df['failure']
print('Normalized class distribution:\n',y.value_counts(normalize=True))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
ratio=(y_train==0).sum()/(y_train==1).sum()
models={'logreg':LogisticRegression(max_iter=1000,class_weight='balanced'),'rf':RandomForestClassifier(n_estimators=300,class_weight='balanced',random_state=42,n_jobs=-1),'xgb':XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.1,scale_pos_weight=ratio,eval_metric='logloss',random_state=42)}
os.makedirs('models',exist_ok=True); rows=[]; probs={}
for name,model in models.items():
 pre=ColumnTransformer([('num',StandardScaler(),NUM),('cat',OneHotEncoder(handle_unknown='ignore'),CAT)])
 pipe=Pipeline([('pre',pre),('model',model)])
 t=time.time(); pipe.fit(X_train,y_train); train_s=time.time()-t
 t=time.time(); p=pipe.predict_proba(X_test)[:,1]; pred=(p>=.5).astype(int); pred_ms=(time.time()-t)*1000
 joblib.dump(pipe,f'models/{name}.joblib'); size=os.path.getsize(f'models/{name}.joblib')/1024
 tn,fp,fn,tp=confusion_matrix(y_test,pred).ravel(); probs[name]=p
 rows.append({'Model':name,'Accuracy':accuracy_score(y_test,pred),'Precision':precision_score(y_test,pred),'Recall':recall_score(y_test,pred),'F1':f1_score(y_test,pred),'ROC_AUC':roc_auc_score(y_test,p),'PR_AUC':average_precision_score(y_test,p),'FN':fn,'FP':fp,'Train_s':train_s,'Predict_ms':pred_ms,'Size_KB':size})
res=pd.DataFrame(rows); res.to_csv('model_comparison.csv',index=False); print('\n',res.to_string(index=False))
# Five genuine held-out test rows for API verification
five=X_test.head(5).copy(); five['actual_label']=y_test.loc[five.index].values; five.to_csv('five_real_test_rows.csv',index=False)
# Threshold experiment on best model by F1/PR-AUC (XGBoost for this run)
best=res.sort_values(['F1','PR_AUC'],ascending=False).iloc[0]['Model']; out=[]
for th in [.5,.3]:
 pred=(probs[best]>=th).astype(int); tn,fp,fn,tp=confusion_matrix(y_test,pred).ravel()
 out.append({'model':best,'threshold':th,'recall':recall_score(y_test,pred),'precision':precision_score(y_test,pred),'f1':f1_score(y_test,pred),'FN':fn,'FP':fp})
pd.DataFrame(out).to_csv('threshold_comparison.csv',index=False)
# independent reload proof
loaded=joblib.load('models/xgb.joblib'); sample=pd.DataFrame([{'type':'L','air_temp':302.0,'process_temp':310.0,'rpm':1300,'torque':65.0,'tool_wear':220}]); print('\nReloaded XGBoost probability:',loaded.predict_proba(sample)[0].tolist())
