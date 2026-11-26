import requests, pandas as pd
URL='http://127.0.0.1:8000'
healthy={'type':'M','air_temp':298.5,'process_temp':309.0,'rpm':1550,'torque':38.0,'tool_wear':20}
risky={'type':'L','air_temp':302.0,'process_temp':310.0,'rpm':1300,'torque':65.0,'tool_wear':220}
def show(name,r): print(f'{name}: HTTP {r.status_code} -> {r.json()}')
show('1 health',requests.get(f'{URL}/health'))
show('2 healthy compare',requests.post(f'{URL}/compare',json=healthy))
show('3 risky compare',requests.post(f'{URL}/compare',json=risky))
show('4 invalid type',requests.post(f'{URL}/compare',json={**healthy,'type':'X'}))
missing=healthy.copy(); missing.pop('torque'); show('5 missing torque',requests.post(f'{URL}/compare',json=missing))
show('6 negative rpm',requests.post(f'{URL}/compare',json={**healthy,'rpm':-100}))
show('7 unknown model',requests.post(f'{URL}/predict/svm',json=healthy))
rows=pd.read_csv('five_real_test_rows.csv'); hits={m:0 for m in ['logreg','rf','xgb']}
print('\n8 five genuine held-out rows:')
for i,row in rows.iterrows():
 actual=int(row['actual_label']); payload={k:row[k].item() if hasattr(row[k],'item') else row[k] for k in ['type','air_temp','process_temp','rpm','torque','tool_wear']}
 r=requests.post(f'{URL}/compare',json=payload); print('row',i+1,'HTTP',r.status_code,'actual',actual,r.json())
 for x in r.json(): hits[x['model']]+=int(x['prediction']==actual)
print('Agreement:',{m:f'{v}/5 ({v/5:.0%})' for m,v in hits.items()})
print('\n100-request average model latency:')
for m in ['logreg','rf','xgb']:
 vals=[requests.post(f'{URL}/predict/{m}',json=risky).json()['latency_ms'] for _ in range(100)]
 print(m,round(sum(vals)/len(vals),4),'ms')
