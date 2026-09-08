# Smart Demand Forecasting and Shipment Delay Prediction

A local web application for inventory demand forecasting and shipment-delay prediction. The project combines a static HTML dashboard with a FastAPI backend that reads CSV data, forecasts product demand, and classifies whether a shipment is likely to be delayed.

## Features

- Product history and summary statistics
- Daily demand forecasting with SARIMAX/ARIMA
- Optional Prophet forecasting when `prophet` is installed
- Forecast confidence intervals and CSV/PNG exports
- Shipment-delay model training with a Random Forest classifier
- Shipment-delay prediction with categorical and numeric features
- Interactive API documentation through FastAPI Swagger UI

## Project Files

| File | Purpose |
| --- | --- |
| `index.html` | Forecast and shipment-delay dashboard |
| `landing.html` | Application landing page |
| `login.html` | Dashboard login page |
| `ml_backend.py` | FastAPI backend and machine-learning endpoints |
| `ProjectMLDataSet.csv` | Product demand history |
| `ShipmentDelays.csv` | Shipment-delay training data |
| `ML_ProjectPPT.pdf` | Project presentation |
| `Project_report.pdf` | Project report |

## Requirements

- Python 3.10 or newer
- A modern web browser
- Git

The backend uses FastAPI, Uvicorn, pandas, NumPy, statsmodels, scikit-learn, and matplotlib. Prophet is optional and is required only for the Prophet forecast option.

## Clone and Run

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Sanchitk99/Smart-Demand-Forecasting-and-Shipment-Delay-Prediction.git
cd Smart-Demand-Forecasting-and-Shipment-Delay-Prediction
```

Create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, allow it for the current terminal session and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the backend dependencies:

```bash
python -m pip install fastapi uvicorn pandas numpy statsmodels scikit-learn matplotlib
```

Start the backend in the first terminal:

```bash
python -m uvicorn ml_backend:app --host 127.0.0.1 --port 8000 --reload
```

Start the frontend in a second terminal. Activate the virtual environment in that terminal first, then run:

```bash
python -m http.server 5500 --bind 127.0.0.1
```

Open the application at [http://127.0.0.1:5500/landing.html](http://127.0.0.1:5500/landing.html). The dashboard is available at [http://127.0.0.1:5500/index.html](http://127.0.0.1:5500/index.html), and API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

Stop either server with `Ctrl+C`.

## Quick Start (Existing Checkout)

Open PowerShell in the project directory:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn ml_backend:app --host 127.0.0.1 --port 8000 --reload
```

In a second PowerShell window, serve the frontend:

```powershell
python -m http.server 5500 --bind 127.0.0.1
```

Open the application at [http://127.0.0.1:5500/landing.html](http://127.0.0.1:5500/landing.html). The dashboard is available at [http://127.0.0.1:5500/index.html](http://127.0.0.1:5500/index.html), and API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

The HTML dashboard expects the API at `http://127.0.0.1:8000`.

## Troubleshooting

### `No module named uvicorn`

This usually means the virtual environment is not active or dependencies were installed into a different Python installation. Activate `.venv` and install/run Uvicorn through the same interpreter:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install uvicorn
python -m uvicorn ml_backend:app --host 127.0.0.1 --port 8000
```

### PowerShell activation is blocked

Run this once in the current PowerShell window, then activate `.venv`:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

## API Endpoints

### Demand forecasting

- `GET /products` - list available products and categories
- `GET /history?product=<name>` - return historical daily demand
- `GET /summary?product=<name>` - return average, minimum, maximum, and trend
- `GET /forecast?product=<name>&model=arima&steps=7` - generate a forecast
- `GET /export/forecast_csv` - download forecast data as CSV
- `GET /export/plot_png` - download a forecast plot as PNG
- `POST /retrain?product=<name>` - clear the cached model for a product

### Shipment-delay prediction

- `GET /delay/schema` - view required fields and examples
- `GET /delay/sample` - return sample shipment records
- `POST /delay/train` - train the Random Forest model
- `POST /delay/predict` - predict delay probability for one shipment
- `GET /delay/metrics` - view trained-model feature importance
- `POST /delay/upload` - append shipment records to the CSV dataset

The shipment dataset uses these fields:

`Origin`, `Destination`, `Carrier`, `WeatherCondition`, `DistanceKM`, `ExpectedDeliveryDays`, `Holiday`, `CustomsClearanceTime`, `HandlingTimeWarehouse`, and `Delay`.

## Configuration

The backend reads these environment variables when provided:

- `CSV_PATH` - path to the demand dataset; defaults to `ProjectMLDataSet.csv`
- `DELAY_CSV_PATH` - path to the shipment dataset; defaults to `ShipmentDelays.csv`

Example:

```powershell
$env:CSV_PATH = "ProjectMLDataSet.csv"
$env:DELAY_CSV_PATH = "ShipmentDelays.csv"
```

## Notes

- Train the shipment-delay model with `POST /delay/train` before calling `POST /delay/predict`.
- The backend enables CORS for local frontend access.
- Login and user authentication are currently handled by the static frontend and are not backed by a database or authentication service.
