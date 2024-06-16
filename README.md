# AMHE2024L

### instalation
1. Create Your python 3.10 env (conda, venv)
2. cd AMHE2024L 
3. ```activate env```
4. ```pip install -e .```

### run
```python run.py --agent des --objective almost-twin-peaks -n 10```
```python run.py --agent ls-shade-rsp --objective quad -n 10```
```python run.py --agent hybridization --objective cec-1 -n 10```

### visualization
```python lookat.py --agent des --objective almost-twin-peaks --history src/checkpoints/lastDES.npy -n 10```

### run benchmark for algorithm 
```python test_cec.py -a des```