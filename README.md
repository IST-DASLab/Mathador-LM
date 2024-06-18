# Mathador-LM: A Dynamic Benchmark for Mathematical Reasoning on Large Language Models

## Environment Setup
1. ```conda create --name mathador python=3.11 -y```
2. ```conda activate mathador```
3. ```pip install -r requirements.txt```
4. Get your personal API key for any of the following providers: OpenAI, TogetherAI, Anthropic.
5. Open `eval.yaml` and configure which models to evaluate. We provide examples for all three model providers.

## Usage

For convenience, we attach `mathador-10000.jsonl` dataset that we used for some runs.
If you would like to generate a new instance of the dataset, please configure `generate_dataset.yaml` and run:
```
python generate_dataset.py generate_dataset.yaml
```

To run Mathador-LM benchmark, please specify your desired parameters in `eval.yaml` and run:
```
TOGETHER_API_KEY=<your_key> python eval.py eval.yaml
```

If you would like to override arguments from `eval.yaml` directly from command-line, please use:
```
TOGETHER_API_KEY=<your_key> python eval.py eval.yaml shots=20
```

The result of the evaluation will be saved in `results.csv`.
