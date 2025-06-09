# AMHE2024L
A hybrid differential evolution algorithm combining NL-SHADE-RSP and DES was developed to enhance optimization performance. It integrates adaptive mutation and nonlinear population reduction with directional momentum and noise-driven exploration. The hybrid approach significantly reduces execution time while maintaining or improving solution quality on CEC 2022 benchmarks.

# Use 
### instalation
1. Create Your python 3.10 env (conda, venv)
2. cd AMHE2024L 
3. ```activate env```
4. ```pip install -e ```

### run
```python run.py --agent des --objective almost-twin-peaks```
```python run.py --agent ls-shade-rsp --objective quad```

### visualization
```python lookat.py --agent des --objective almost-twin-peaks --history src/checkpoints/lastDES.npy```

# Documentation (only in polish, sorry)
Check NES_NL_SHADE_hybridization_documentation.pdf file
