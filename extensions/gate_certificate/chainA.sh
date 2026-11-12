set -x
python3 null_pool.py case9 5000 0.08 nominal
python3 null_pool.py case9 500 0.08 noise3 0.003
python3 null_pool.py case9 500 0.08 inert110 0.001 1.10
python3 null_pool.py case9 500 0.08 inert105 0.001 1.05
python3 null_pool.py case9 500 0.16 load16 0.001
python3 null_pool.py case39 2500 0.08 nominal
python3 null_pool.py case39 400 0.08 noise3 0.003
python3 null_pool.py case39 400 0.08 inert110 0.001 1.10
python3 null_pool.py case39 400 0.08 inert105 0.001 1.05
python3 null_pool.py case39 400 0.16 load16 0.001
