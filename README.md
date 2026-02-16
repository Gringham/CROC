# CROC <br><sub><sup>CROC: Evaluating and Training T2I Metrics with Pseudo- and Human-Labeled Contrastive Robustness Checks</sup></sub>

In this repository, we release the source code of our dataset CROC and the metric CROCScore. For usage instructions, please see the respective folder's readme files, as well as the main part of the scripts and comments in the code. 

[![📄 arXiv](https://img.shields.io/badge/View%20on%20arXiv-B31B1B?logo=arxiv&labelColor=gray)](https://arxiv.org/abs/2505.11314)

## 📁 Repository Structure

```
├── croc_hum            # Everything related to CROChum, our human supervised dataset, i.e., scripts for data generation, evaluation and human evaluation
├── croc_syn            # Everything related to CROCsyn, our synthetic dataset, i.e., scripts for data generation, evaluation and human evaluation
├── metrics             # Wrapper scripts to apply T2I metrics including CROCScore
├── pyproject.toml      # List of poetry dependencies
├── README.md           # This Project documentation
└── ...
```

---

## 🛠️ Setup & Installation

To set up the environment and install dependencies:

```bash
pip install uv

git clone https://github.com/Gringham/CROC.git
cd CROC

uv venv --python 3.11
uv pip install -U timm  flash-attn==2.7.3 vllm transformers==4.49 diffusers[torch]==0.34.0 --no-build-isolation --no-cache-dir 
#Note that some parts of CROC have conflicting dependencies and may require different environments.
```

After installing the dependencies, please set your cache directory in the following locations
```
- project_root.py
- croc_hum/metric_apply.sh
- croc_syn/img_gen/img_gen.sh
- further_benchmarks/3_genaibench_experiment/t2v_metrics/genai_image_eval_customscore.py
- metrics/apply_metrics.sh
- metrics/VQAScore.py
```

In all slurm scripts, specify the activation of your specific environment. 

---

## 🚀 Usage

Please view the readme files in the respective subfolders.
---

## 📌 Notice on AI Generation

Some parts of the code were written with AI support by Github Copilot and GPT-o3/o4.

---

## 📖 Citation

If you use this work in your research, please cite it as:

```bibtex
@misc{leiter2025crocevaluatingtrainingt2i,
      title={CROC: Evaluating and Training T2I Metrics with Pseudo- and Human-Labeled Contrastive Robustness Checks}, 
      author={Christoph Leiter and Yuki M. Asano and Margret Keuper and Steffen Eger},
      year={2025},
      eprint={2505.11314},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.11314}, 
}
```

---