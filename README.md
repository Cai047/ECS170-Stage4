# ECS 170 Stage 4 · Text generation

Person 4's contribution compares a vanilla RNN, LSTM and GRU on the instructor's text generation corpus. The companion text classification task can be added alongside generation in the same course template.

| Directory | Contents |
| --- | --- |
| `local_code/base_class/` | Instructor-provided base classes |
| `local_code/stage_4_code/generation/` | Corpus preparation, recurrent language models, training and evaluation |
| `script/stage_4_script/` | Generation experiment entry points |
| `results/generation/` | Measured learning curves, metrics and generated text |
| `report/` | Editable generation section for the joint Stage 4 report |

## Reproduce the generation experiment

Tested on CPU with Python 3.12.6. Install the dependencies below in a Python 3.12 environment. Download the [instructor's Stage 4 dataset](https://drive.google.com/file/d/1yNr8deq1JO-HNhlAa1yV4Zycjqmr1Hfz/view) and extract its `stage_4_data` folder into `data/`. The generation file is named `data` with no extension and contains the columns `ID` and `Joke`.

```sh
python -m pip install -r requirements.txt
COURSE_DATA=data/stage_4_data/text_generation/data
python -m script.stage_4_script.replay_generation --data "$COURSE_DATA"
python -m unittest discover -s tests -v
```

The replay command uses the three included CPU checkpoints to verify the final test scores and all 27 samples. To train and evaluate a fresh set of models, run from the repository root:

```sh
python -m script.stage_4_script.train_generation --data "$COURSE_DATA" \
  --prompts "what did the" "why did the" "how do you" --epochs 20 \
  --output results_new/generation
python -m script.stage_4_script.evaluate_generation --data "$COURSE_DATA" \
  --results-dir results_new/generation
```

All runs use the course jokes and word embeddings trained with the language model. No external embeddings are needed. The code groups duplicate normalized jokes before partitioning with seed 170. It fits the vocabulary on 1,299 training jokes, checks learning curves on 160 validation jokes, and evaluates the fixed final epoch on 163 test jokes. Test perplexity includes the unknown-word class and the end of each joke. Compare it across these models using the same dataset and vocabulary.

| Cell | Parameters | Test perplexity ↓ | Next-token accuracy |
| --- | ---: | ---: | ---: |
| RNN | 193,181 | 47.86 | 30.25% |
| LSTM | 215,069 | 41.30 | 32.16% |
| GRU | 207,773 | 44.43 | 31.58% |

[Learning curves](results/generation/learning_curves.png) show the training and validation losses. [Evaluation details](results/generation/evaluation.json) pair generated examples with training references; [per-prompt comparisons](results/generation/summary_by_prompt.csv) pool the three sampling seeds for each cell and prompt. Each model folder contains its unedited samples, epoch logs, experiment settings and checkpoint. For the shared project, the classification contributor can add a sibling `local_code/stage_4_code/classification/` and their own scripts and results.
