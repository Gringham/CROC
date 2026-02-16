#PYTHONPATH=$(pwd) python further_benchmarks/1_winoground.py --metric alignscore
#PYTHONPATH=$(pwd) python further_benchmarks/1_winoground.py --metric vqascore
PYTHONPATH=$(pwd) python further_benchmarks/1_winoground.py --metric crocscore
PYTHONPATH=$(pwd) python further_benchmarks/1_winoground.py --metric phiscore
#PYTHONPATH=$(pwd) python further_benchmarks/1_winoground.py --metric blip2itm