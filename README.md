A streamlit app for classification graient boosting model

# To download, build and run 
## On Windows
1. `python -m venv .\venv`
2. `.\venv\Scripts\Activate.ps1`
3. `pip install -e .`
4. `streamlit run .\src\course_ml_streamlit\main.py`

# To install dev dependencies run in venv 
- `pip install -e ".[testing]"`

Crashes are likely caused by LightGBM, refer to this [FAQ](https://lightgbm.readthedocs.io/en/latest/FAQ.html#i-encounter-segmentation-faults-segfaults-randomly-after-installing-lightgbm-from-pypi-using-pip-install-lightgbm)
