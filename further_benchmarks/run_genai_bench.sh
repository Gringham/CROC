#PYTHONPATH=$(pwd) python further_benchmarks/3_genaibench_experiment/t2v_metrics/genai_image_eval_customscore.py --model alignscore
#PYTHONPATH=$(pwd) python further_benchmarks/3_genaibench_experiment/t2v_metrics/genai_image_eval_customscore.py --model vqascore
PYTHONPATH=$(pwd) python further_benchmarks/3_genaibench_experiment/t2v_metrics/genai_image_eval_customscore.py --model crocscore
PYTHONPATH=$(pwd) python further_benchmarks/3_genaibench_experiment/t2v_metrics/genai_image_eval_customscore.py --model phiscore
PYTHONPATH=$(pwd) python further_benchmarks/3_genaibench_experiment/t2v_metrics/genai_image_eval_customscore.py --model blip2itm