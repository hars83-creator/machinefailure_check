import time, joblib, pandas as pd
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
app=FastAPI(title="Machine Failure Prediction API")
MODELS={name:joblib.load(f"models/{name}.joblib") for name in ["logreg","rf","xgb"]}
class Machine(BaseModel):
    type: Literal["L","M","H"]
    air_temp: float
    process_temp: float
    rpm: float=Field(...,gt=0)
    torque: float=Field(...,ge=0)
    tool_wear: float=Field(...,ge=0)
def run(name,m):
    X=pd.DataFrame([m.model_dump()]); t0=time.perf_counter(); p=float(MODELS[name].predict_proba(X)[0,1]); ms=(time.perf_counter()-t0)*1000
    return {"model":name,"failure_probability":p,"prediction":int(p>=0.5),"latency_ms":ms}
@app.get("/health")
def health(): return {"status":"ok","models":list(MODELS.keys())}
@app.post("/predict/{model_name}")
def predict(model_name:str,m:Machine):
    if model_name not in MODELS: raise HTTPException(status_code=404,detail=f"Unknown model '{model_name}'. Choose logreg, rf, or xgb.")
    return run(model_name,m)
@app.post("/compare")
def compare(m:Machine): return [run(name,m) for name in MODELS]
