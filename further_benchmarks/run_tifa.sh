#PYTHONPATH=$(pwd) python further_benchmarks/2_tifa.py --metric alignscore
#PYTHONPATH=$(pwd) python further_benchmarks/2_tifa.py --metric vqascore
#PYTHONPATH=$(pwd) python further_benchmarks/2_tifa.py --metric crocscore
PYTHONPATH=$(pwd) python further_benchmarks/2_tifa.py --metric phiscore
#PYTHONPATH=$(pwd) python further_benchmarks/2_tifa.py --metric blip2itm