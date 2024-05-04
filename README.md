# AMHE2024L

### instalation
1. Create Your python 3.9 env (conda, venv)
2. ```cd AMHE2024L``` 
3. ```activate env```
4. ```pip install -e ```

### run

```bash
python run.py --agent des --family simple --objective almost-twin-peaks
```

```bash
python run.py --a ls-shade-rsp -f cec -o 12
```

### visualization

```bash
python lookat.py --agent des --family simple --objective almost-twin-peaks --history src/checkpoints/lastDES.npy -i 1
```