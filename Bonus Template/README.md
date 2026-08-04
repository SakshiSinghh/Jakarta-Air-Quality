# Jakarta AQI Streamlit Bonus

## Structure

```text
Bonus Template/
├── Home.py
├── library.py
├── requirements.txt
├── aqi_model.npz
├── jakarta_air_quality_cleaned.xlsx
└── pages/
    ├── 1_Model_and_Methodology.py
    └── 2_Dataset_Transparency.py
```

## Run

```bash
python -m pip install -r requirements.txt
python -m streamlit run Home.py
```

Place `aqi_model.npz` in the same folder as `Home.py` before running the app.
