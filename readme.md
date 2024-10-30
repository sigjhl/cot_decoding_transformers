# Chain of Thought Decoding for Transformers

This project implements a Chain of Thought (CoT) decoding method ([Paper](https://arxiv.org/abs/2402.10200)) for transformer models using PyTorch and Hugging Face's Transformers library. The CoT approach enhances the reasoning capabilities of models by allowing them to generate intermediate steps in their thought process.


## Features

- Load various transformer models and tokenizers from Hugging Face Hub.
- Calculate CoT scores to assess the quality of generated answers.
- Supports both sequential and parallel decoding (slightly less VRAM usage for sequential mode at the cost of being much slower).
- Optimized for both CPU and GPU (CUDA and MPS) environments.
- Configurable model selection, question input, and decoding parameters.
- Compatible with Llama, Phi models (Gemma2 is not working at the moment).

## Installation

To set up the environment, you can create a new virtual environment and install the required packages using the provided `requirements.txt`.

```bash
# Create a new environment (optional)
conda create -n cot-decoding python=3.10
conda activate cot-decoding
```

Then, install the requirements:

```bash
pip install -r requirements.txt
```

## Usage

Run the script with the following command:

```bash
python main.py --model_name <model_name> --question "<your_question>" --k <number_of_branches> --aggregation <max|sum> --device <cuda|cpu|mps>
```

### Example

```bash
python main.py --model_name "meta-llama/Llama-3.2-1B-Instruct" --question "Sally has two brothers, Sam and Joe. Sam has one sister. How many sisters does Joe have? Think step by step. You MUST end your reply with Answer:, FOLLOWED BY A SINGLE NUMBER." --k 10 --aggregation max --device cuda
```

## Arguments

- `--model_name`: Model checkpoint name (default: `meta-llama/Llama-3.2-1B-Instruct`).
- `--question`: Question to ask the model (default: "Sally has two brothers...").
- `--k`: Number of decoding branches (default: 10).
- `--aggregation`: Method for aggregating CoT scores (`max` or `sum`, default: `max`).
- `--device`: Device to run the model on (`cuda`, `cpu`, or `mps`, default: `cuda`).
- `--use_sequential`: Use sequential processing for low RAM situations (optional).
- `--system_prompt`: Use a custom system prompt. If not given, defaults to the chat template included in the tokenizer (optional).

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
