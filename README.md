# Text-to-SQL with LoRA / QLoRA

Fine-tuning small open LLMs to turn a plain-English question plus a table schema into a SQL query.
Built as a hands-on project to learn LoRA and QLoRA on a single Colab T4 GPU.

Author: Sachiin M

## Results

300 held-out test questions. Exact match = same SQL as the reference after normalizing case and whitespace.
"Runs on SQLite" = the query executes without error on empty tables built from the schema.

| Model | Setup | Exact match | Runs on SQLite |
|---|---|---|---|
| Qwen2.5-Coder-1.5B | untuned | 49.7% | 95.0% |
| Qwen2.5-Coder-1.5B | LoRA r=16, lr 2e-4 | 81.3% | 98.0% |
| Qwen2.5-Coder-1.5B | LoRA r=4, lr 2e-4 | 81.0% | 97.7% |
| Qwen2.5-Coder-1.5B | LoRA r=64, lr 2e-4 | 82.3% | 97.3% |
| Qwen2.5-Coder-1.5B | LoRA r=16, lr 5e-5 | 78.3% | 97.7% |
| Qwen2.5-Coder-7B (4-bit) | untuned | 58.0% | 97.0% |
| Qwen2.5-Coder-7B (4-bit) | QLoRA r=16, lr 2e-4, 4,000 examples | 77.3% | 97.3% |

## Findings

- LoRA lifted the 1.5B model's exact match from 49.7% to 81.3%. It fixed 103 test questions and broke 8.
- The untuned model already wrote valid SQL (95% ran). What it lacked was the dataset's conventions. Fine-tuning mostly teaches style and format, not new knowledge.
- Rank barely mattered: r=4 (4.6M trainable parameters) scored within about 1 point of r=64 (73.9M), well inside the test noise. Validation loss did improve slightly with rank.
- A lower learning rate (5e-5) was worse after one epoch, so it was probably under-trained.
- QLoRA on the 7B model did not beat the 1.5B LoRA model (77.3% vs 81.3%). This is not a clean comparison: the 7B used half the training data, 4-bit weights and untuned hyperparameters.
- Reading the errors: about two thirds of the remaining "mistakes" were equivalent SQL (for example `week = 4` vs `week = "4"`, swapped table aliases, JOIN vs IN). The real accuracy is likely higher than exact match shows.
- Some references in the dataset are malformed (for example `13,575` split into `13 OFFSET 575`), and the model learned those quirks.

## Limitations

- 300 test questions gives a margin of error of about plus or minus 4.5 points. Single runs, one random seed.
- Exact match is strict. "Runs on SQLite" is lenient: queries run on empty tables, so a wrong query can still pass.
- The dataset is mostly simple, single-table questions. These results do not show general SQL ability.

## Setup

- Data: b-mc2/sql-create-context from Hugging Face (8,000 train / 200 validation / 300 test, fixed seed).
- Training: Hugging Face Transformers + PEFT, 1 epoch, effective batch size 16, cosine schedule, fp16, LoRA on all attention and MLP projections (alpha = 2 x rank). The loss is computed only on the SQL answer, not the prompt.
- Hardware: Google Colab T4 (15 GB).

## Notebooks (run in order)

1. `01_data.ipynb`: load the dataset, split it, build the prompt format.
2. `02_baseline.ipynb`: score the untuned model and create `utils.py` (scoring helpers).
3. `03_lora_train_eval.ipynb`: train and evaluate LoRA r=16.
4. `04_experiments.ipynb`: rank and learning-rate experiments.
5. `05_qlora_7b.ipynb`: QLoRA on the 7B model.

The notebooks save data, results and adapters to a `text2sql-lora` folder on Google Drive.
