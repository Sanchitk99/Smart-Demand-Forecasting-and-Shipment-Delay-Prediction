import os
import io
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", ConvergenceWarning)

# ============================================================
# FIRST FIX: CREATE FASTAPI APP AT TOP BEFORE ANY DECORATORS
# ============================================================

app = FastAPI(title="Inventory Forecast API (Option A)", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
# ============================================================
# ✅ SHIPMENT DELAY PREDICTION MODULE (robust + self-contained)
# ============================================================
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

DELAY_MODEL = None
DELAY_ENCODERS: Dict[str, LabelEncoder] = {}
DELAY_REQUIRED = ["Origin","Destination","Carrier",
    "WeatherCondition","DistanceKM","ExpectedDeliveryDays","Holiday","CustomsClearanceTime","HandlingTimeWarehouse","Delay"]
DELAY_CSV_PATH = os.environ.get("DELAY_CSV_PATH", "ShipmentDelays.csv")

def _ensure_delay_csv(path: str = DELAY_CSV_PATH):
    """Create a tiny synthetic dataset if file is missing, so /delay/train always works."""
    if os.path.exists(path):
        return
    rng = np.random.default_rng(42)
    origins = ["Delhi","Mumbai","Chennai","Bengaluru","Kolkata"]
    dests   = ["Pune","Hyderabad","Jaipur","Ahmedabad","Lucknow"]
    carriers= ["BlueDart","Delhivery","DTDC","FedEx","UPS"]
    weather = ["Sunny","Cloudy","Rainy","Storm","Fog"]
    rows = []
    for _ in range(400):
        o = rng.choice(origins); d = rng.choice(dests); c = rng.choice(carriers); w = rng.choice(weather)
        dist = int(rng.integers(50, 2000))
        etd  = int(rng.integers(2, 10))                              # expected days
        hol  = int(rng.integers(0, 2))                               # holiday/promo flag
        customs = float(rng.uniform(0.2, 3.5))                       # days
        handling = float(rng.uniform(0.1, 2.0))                      # days
        # crude rule to simulate delays:
        delay_score = (dist/1500) + (customs*0.5) + (handling*0.4) + (1 if w in ["Storm","Fog"] else 0) + (0.3 if hol else 0)
        delayed = int(delay_score > 1.8)
        rows.append([o,d,c,w,dist,etd,hol,customs,handling,delayed])
    df = pd.DataFrame(rows, columns=DELAY_REQUIRED)
    df.to_csv(path, index=False)

def load_delay_dataset(path: str = DELAY_CSV_PATH) -> pd.DataFrame:
    _ensure_delay_csv(path)
    df = pd.read_csv(path)
    for r in DELAY_REQUIRED:
        if r not in df.columns:
            raise ValueError(f"Shipment delay dataset missing column: {r}")
    return df

def _fit_labelencoder_with_unknown(series: pd.Series) -> LabelEncoder:
    """Fit LabelEncoder and make sure 'UNK' exists for unseen categories at predict time."""
    le = LabelEncoder()
    vals = series.astype(str).fillna("UNK").tolist()
    if "UNK" not in vals:
        vals = vals + ["UNK"]
    le.fit(vals)
    return le

def _encode_with_unknown(le: LabelEncoder, vals: pd.Series) -> np.ndarray:
    """Map unseen categories to 'UNK'"""
    classes = set(le.classes_.tolist())
    safe = vals.astype(str).apply(lambda v: v if v in classes else "UNK")
    return le.transform(safe)

def prepare_delay_data(df: pd.DataFrame):
    df = df.copy()
    # categorical columns to encode
    cat_cols = ["Origin","Destination","Carrier","WeatherCondition","Holiday"]
    # fit encoders
    global DELAY_ENCODERS
    DELAY_ENCODERS = {}
    for c in cat_cols:
        DELAY_ENCODERS[c] = _fit_labelencoder_with_unknown(df[c])

    X = pd.DataFrame()
    for c in cat_cols:
        X[c] = _encode_with_unknown(DELAY_ENCODERS[c], df[c])

    # numeric features
    num_cols = ["DistanceKM","ExpectedDeliveryDays","CustomsClearanceTime","HandlingTimeWarehouse"]
    for c in num_cols:
        X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    y = df["Delay"].astype(int)
    return X, y

@app.get("/delay/schema")
def delay_schema():
    """Returns required columns and types."""
    return {
        "required_columns": DELAY_REQUIRED,
        "categorical": ["Origin","Destination","Carrier","WeatherCondition","Holiday"],
        "numeric": ["DistanceKM","ExpectedDeliveryDays","CustomsClearanceTime","HandlingTimeWarehouse"],
        "target": "Delay",
        "examples_endpoint": "/delay/sample",
        "train_endpoint": "/delay/train",
        "predict_endpoint": "/delay/predict"
    }

@app.get("/delay/sample")
def delay_sample(n: int = 5):
    df = load_delay_dataset()
    return {"rows": df.sample(min(n, len(df)), random_state=1).to_dict(orient="records")}

@app.post("/delay/upload")
def delay_upload(rows: List[Dict]):
    """Append rows to the CSV (simple ingestion)."""
    if not rows:
        raise HTTPException(400, "No rows provided.")
    df = load_delay_dataset()
    new = pd.DataFrame(rows)
    missing = [c for c in DELAY_REQUIRED if c not in new.columns]
    if missing:
        raise HTTPException(400, f"Missing columns in payload: {missing}")
    df = pd.concat([df, new[DELAY_REQUIRED]], ignore_index=True)
    df.to_csv(DELAY_CSV_PATH, index=False)
    return {"status": "ok", "rows_after": len(df)}

@app.post("/delay/train")
def train_delay_model(path: str = "ShipmentDelays.csv"):
    global DELAY_MODEL

    df = load_delay_dataset(path)
    X, y = prepare_delay_data(df)

    # Prevent train/test class imbalance errors
    if len(set(y)) < 2:
        return {
            "status": "trained",
            "accuracy": 1.0,
            "f1_score": 0.0,
            "roc_auc": 0.0,
            "note": "Dataset has only one class — F1 & ROC-AUC not applicable."
        }

    # ✅ Use stratified splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    DELAY_MODEL = model

    # ---- Safe Metrics ----
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    acc = accuracy_score(y_test, pred)

    try:
        f1 = f1_score(y_test, pred)
    except:
        f1 = 0.0

    try:
        roc = roc_auc_score(y_test, prob)
    except:
        roc = 0.0

    return {
        "status": "trained",
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": roc
    }

@app.post("/delay/predict")
def delay_predict(shipment: Dict):
    if DELAY_MODEL is None:
        raise HTTPException(400, "Delay model not trained. Call /delay/train first.")
    # turn dict -> DataFrame
    df = pd.DataFrame([shipment])

    # sanity + defaults
    for col in DELAY_REQUIRED:
        if col not in df.columns:
            # for target (Delay) ignore; for others set defaults
            if col == "Delay":
                continue
            # sensible defaults for quick testing
            if col in ["Origin","Destination","Carrier","WeatherCondition"]:
                df[col] = "UNK"
            elif col == "Holiday":
                df[col] = 0
            elif col in ["DistanceKM","ExpectedDeliveryDays"]:
                df[col] = 1
            elif col in ["CustomsClearanceTime","HandlingTimeWarehouse"]:
                df[col] = 0.0

    # build feature tensor the same way as training
    X = pd.DataFrame()
    cat_cols = ["Origin","Destination","Carrier","WeatherCondition","Holiday"]
    num_cols = ["DistanceKM","ExpectedDeliveryDays","CustomsClearanceTime","HandlingTimeWarehouse"]

    for c in cat_cols:
        le = DELAY_ENCODERS.get(c)
        if le is None:
            raise HTTPException(500, f"Encoder not found for {c}. Train again.")
        X[c] = _encode_with_unknown(le, df[c].astype(str))

    for c in num_cols:
        X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    proba = float(DELAY_MODEL.predict_proba(X)[0][1])
    pred = int(proba >= 0.5)
    return {
        "predicted_delay": pred,
        "delay_probability": proba,
        "features_used": X.iloc[0].to_dict()
    }

@app.get("/delay/metrics")
def delay_metrics():
    """Feature importances (if model is trained)."""
    if DELAY_MODEL is None:
        raise HTTPException(400, "Delay model not trained.")
    # infer feature names from encoders order used in prepare
    feat_names = ["Origin","Destination","Carrier","WeatherCondition","Holiday",
                  "DistanceKM","ExpectedDeliveryDays","CustomsClearanceTime","HandlingTimeWarehouse"]
    importances = DELAY_MODEL.feature_importances_
    return {"feature_importances": dict(zip(feat_names, map(float, importances)))}


# ============================================================
# Prophet optional
# ============================================================
try:
    from prophet import Prophet
    HAVE_PROPHET = True
except:
    HAVE_PROPHET = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# REAL CATEGORY STORAGE
# ============================================================
PRODUCT_CATEGORY: Dict[str, str] = {}


# ============================================================
# CONFIG
# ============================================================
CSV_PATH = os.environ.get("CSV_PATH", "ProjectMLDataSet.csv")


# ============================================================
# COLUMN DETECTION
# ============================================================
def _norm_cols(cols: List[str]) -> List[str]:
    return [c.strip().lower().replace(" ", "").replace("_", "") for c in cols]


def _detect_cols(df: pd.DataFrame) -> Tuple[str, str, Optional[str], Optional[str]]:
    cols = df.columns.tolist()
    norm = _norm_cols(cols)

    date_cands = {"date","orderdate","salesdate","day","timestamp"}
    units_cands = {"unitssold","units","quantity","qty","sales","demand"}
    prod_cands  = {"product","productid","skuid","sku","itemid","item"}
    cat_cands   = {"category","productcategory"}

    date_col = next((cols[i] for i,c in enumerate(norm) if c in date_cands), cols[0])

    units_col = None
    for i,c in enumerate(norm):
        if cols[i] != date_col and c in units_cands:
            units_col = cols[i]
            break

    if units_col is None:
        for i in range(len(cols)):
            if cols[i] != date_col:
                try:
                    pd.to_numeric(df.iloc[:,i])
                    units_col = cols[i]
                    break
                except:
                    pass

    if units_col is None:
        raise ValueError("No numeric units column found.")

    product_col = next((cols[i] for i,c in enumerate(norm) if c in prod_cands), None)
    category_col = next((cols[i] for i,c in enumerate(norm) if c in cat_cands), None)

    return date_col, units_col, product_col, category_col


# ============================================================
# LOADING CSV + CATEGORY MAP
# ============================================================
def load_all() -> pd.DataFrame:

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    raw = pd.read_csv(CSV_PATH)
    dcol, ucol, pcol, ccol = _detect_cols(raw)

    df = pd.DataFrame({
        "date": pd.to_datetime(raw[dcol], errors="coerce"),
        "units": pd.to_numeric(raw[ucol], errors="coerce"),
        "product": raw[pcol] if pcol else "DEFAULT",
        "category": raw[ccol] if ccol else "Unknown"
    }).dropna(subset=["date","units","product"])

    df["date"] = df["date"].dt.floor("D")

    global PRODUCT_CATEGORY
    PRODUCT_CATEGORY = (
        df.groupby("product")["category"]
        .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else "Unknown")
        .to_dict()
    )

    return df.groupby(["product","date"], as_index=False)["units"].sum()


def get_category(product: str) -> str:
    return PRODUCT_CATEGORY.get(product, "Unknown")


def load_product_series(product: str) -> pd.Series:
    df = load_all()
    if product not in df["product"].unique():
        raise HTTPException(404, f"Product '{product}' not found.")
    sdf = df[df["product"] == product].sort_values("date")
    s = pd.Series(sdf["units"].values, index=pd.DatetimeIndex(sdf["date"]))
    return s.asfreq("D").fillna(0.0)

# ============================================================
# ARIMA + PROPHET FORECASTING
# ============================================================
def fit_best_sarimax(series: pd.Series):
    try:
        m = SARIMAX(series, order=(1,1,1), seasonal_order=(0,1,1,7),
                    enforce_stationarity=False, enforce_invertibility=False)
        return m.fit(method="powell", maxiter=50, disp=False)
    except:
        m = SARIMAX(series, order=(2,1,0), seasonal_order=(0,1,1,7),
                    enforce_stationarity=False, enforce_invertibility=False)
        return m.fit(method="powell", maxiter=30, disp=False)


def forecast_prophet(series: pd.Series, steps: int):
    if not HAVE_PROPHET:
        raise HTTPException(400, "Prophet not installed.")

    df = pd.DataFrame({"ds": series.index, "y": series.values})
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=steps, freq="D", include_history=False)
    out = m.predict(future)

    return (
        pd.to_datetime(out["ds"]),
        out["yhat"].values,
        out["yhat_lower"].values,
        out["yhat_upper"].values
    )

def compute_summary(series: pd.Series, window=90):
    s = series.tail(window)
    x = np.arange(len(s))
    slope = float(np.polyfit(x, s.values, 1)[0]) if len(s) >= 2 else 0.0
    return {
        "average": float(np.mean(s)),
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "trend_slope_per_day": slope
    }

# ============================================================
# CACHE
# ============================================================
@dataclass
class Cache:
    arima: dict
CACHE = Cache(arima={})

def arima_for(product: str, series: pd.Series):
    if product not in CACHE.arima:
        CACHE.arima[product] = fit_best_sarimax(series)
    return CACHE.arima[product]

# ============================================================
# SCHEMAS
# ============================================================
class HistoryResponse(BaseModel):
    dates: List[str]
    values: List[float]
    n: int

class SummaryResponse(BaseModel):
    average: float
    min: float
    max: float
    trend_slope_per_day: float

class ForecastResponse(BaseModel):
    model: str
    product: str
    category: str
    dates: List[str]
    mean: List[float]
    lower: List[float]
    upper: List[float]

# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/products")
def products():
    df = load_all()
    prods = sorted(df["product"].unique().tolist())
    return {
        "products": [
            {"name": p, "category": get_category(p)}
            for p in prods
        ]
    }
@app.get("/history", response_model=HistoryResponse)
def history(product: str, limit: int = 365):
    s = load_product_series(product).tail(limit)
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in s.index],
        "values": list(map(float, s.values)),
        "n": len(s)
    }
@app.get("/summary", response_model=SummaryResponse)
def summary(product: str, window: int = 90):
    return compute_summary(load_product_series(product), window)


@app.get("/forecast", response_model=ForecastResponse)
def forecast(product: str, model: str = "arima", steps: int = 7):
    series = load_product_series(product)
    if model == "arima":
        m = arima_for(product, series)
        fc = m.get_forecast(steps=steps)
        ci = fc.conf_int(alpha=0.05)
        dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=steps)
        mean = fc.predicted_mean.values
        lower = ci.iloc[:, 0].values
        upper = ci.iloc[:, 1].values
    elif model == "prophet":
        dates, mean, lower, upper = forecast_prophet(series, steps)
    else:
        raise HTTPException(400, "Model must be arima or prophet.")
    return {
        "model": model,
        "product": product,
        "category": get_category(product),
        "dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates],
        "mean": [max(0, float(v)) for v in mean],
        "lower": [max(0, float(v)) for v in lower],
        "upper": [max(0, float(v)) for v in upper]
    }
@app.get("/export/forecast_csv")
def export_csv(product: str, model: str = "arima", steps: int = 7):
    fc = forecast(product, model, steps)
    df = pd.DataFrame({
        "date": fc["dates"],
        "mean": fc["mean"],
        "lower": fc["lower"],
        "upper": fc["upper"],
        "product": fc["product"],
        "category": fc["category"],
        "model": fc["model"]
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=forecast.csv"}
    )

@app.get("/export/plot_png")
def export_png(product: str, model: str = "arima", steps: int = 7):
    h = history(product, limit=5000)
    fc = forecast(product, model, steps)
    dates_hist = pd.to_datetime(h["dates"])
    values_hist = h["values"]
    dates_fc = pd.to_datetime(fc["dates"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates_hist, values_hist, label="Historical")
    ax.plot(dates_fc, fc["mean"], label=f"Forecast ({model})")
    ax.fill_between(dates_fc, fc["lower"], fc["upper"], alpha=0.3)
    ax.legend()
    ax.set_title(f"Forecast — {product} [{fc['category']}] ({model.upper()})")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    return StreamingResponse(buf, media_type="image/png")

@app.post("/retrain")
def retrain(product: str):
    s = load_product_series(product)
    CACHE.arima[product] = fit_best_sarimax(s)
    return {"status": "ok", "product": product}

# RUN SERVER

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ml_backend:app", host="0.0.0.0", port=8000, reload=False)