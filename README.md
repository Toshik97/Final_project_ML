# Final_project_ML

## Project Overview
`Final_project_ML` is an end‑to‑end machine‑learning pipeline that trains and serves a **binary‑classification** model for predicting whether a website visit will lead to one of several key conversion events (e.g. `sub_car_claim_click`, `sub_submit_success`).  

The repository contains two main parts:
1. **Model training notebook / scripts** (`model/ML 3.0.ipynb`, `model/ML 3.0.py`) that clean Google Analytics session & hit logs, engineer categorical features, try different classifiers (Logistic Regression, Random Forest with class balancing) and save the best pipeline with `dill`.
2. **FastAPI micro‑service** (`main.py`) that loads the pre‑trained pipeline (`model/Model_downsampling.pkl`) and exposes REST endpoints for health‑check, versioning and online inference.

> **Target**  
> `y = target_action ∈ {0, 1}` — will the session trigger at least one of eight conversion actions?

---
##  Repository Structure
```
├── .gitignore
├── main.py              # FastAPI application
├── ML 3.0.ipynb         # Full exploratory & training workflow
├── model/
│   ├── Model_downsampling.pkl  # Production model ↳ loaded by main.py
│   ├── Model_balanced          # Alternative model from class‑balanced training run
│   ├── ML 3.0.py, ML 4.0.py    # Script versions of the notebook
│   └── catboost_info/, json/   # Auxiliary artefacts/logs
└── data/ (not in repo)         # Place raw GA csv files here: ga_sessions.csv, ga_hits.csv
```

---
##  Quick Start
1. **Clone & create env**
   ```bash
   git clone https://github.com/Toshik97/Final_project_ML.git && cd Final_project_ML
   python -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt  # see below if the file is missing
   ```
2. **Run the API locally**
   ```bash
   uvicorn main:app --reload
   ```
   The service will be available at **`http://localhost:8000`**.

3. **Check endpoints**
   | Route            | Method | Description                                    |
   |------------------|--------|------------------------------------------------|
   | `/status`        | GET    | Returns *"I'm OK"* if the service is healthy   |
   | `/version`       | GET    | Metadata dict attached to the saved model      |
   | `/predict`       | POST   | Returns `{ "pred": true / false }`             |

4. **Example request**
   ```bash
   curl -X POST http://localhost:8000/predict         -H "Content-Type: application/json"         -d '{
              "utm_medium": "cpc",
              "utm_source": "google",
              "utm_campaign": "spring_sale",
              "device_category": "mobile",
              "device_brand": "Samsung",
              "device_browser": "Chrome",
              "geo_country": "Russia",
              "device_os": "Android",
              "geo_city": "Moscow",
              "utm_keyword": "insurance online"
            }'
   ```

---
##  Training / Re‑training
*Place the raw data into the `data/` folder* (not pushed to the repo because of size/privacy):
- `ga_sessions.csv`
- `ga_hits.csv`

Run either the notebook (`model/ML 3.0.ipynb`) or the corresponding script:
```bash
python model/ML\ 3.0.py
```
The script:
1. Merges sessions & hits, removes obvious duplicates and missing values.
2. Engineers categorical features (grouping rare categories, imputation, one‑hot encoding).
3. Balances the target class by **down‑sampling** (production model) or class weights (alternative model).
4. Tries Logistic Regression (L1) & Random Forest and picks the highest 10‑fold CV accuracy.
5. Saves a dictionary with keys `model` (pipeline) and `metadata` (author, version, metrics) via *dill*.

> **Tip**‑ You can tweak model choices and CV metrics inside `model/ML 3.0.py`.

---
##  Requirements
```text
fastapi
uvicorn[standard]
pandas
scikit‑learn>=1.4
numpy
dill
catboost   # optional, only if you experiment further
jupyter     # for notebook users
```


