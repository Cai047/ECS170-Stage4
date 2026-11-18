# ECS 170 Stage 4 · Text generation

Person 4's contribution compares a vanilla RNN, LSTM and GRU on the instructor's text generation corpus. The companion text classification task can be added alongside generation in the same course template.

| Directory | Contents |
| --- | --- |
| `local_code/base_class/` | Instructor-provided base classes |
| `local_code/stage_4_code/generation/` | Corpus preparation, recurrent language models, training and evaluation |
| `script/stage_4_script/` | Generation experiment entry points |
| `results/generation/`, `results/generation_wide/` | Results and weights for the 20-epoch comparison and 45-epoch manual demo |
| `report/` | Editable generation section for the joint Stage 4 report |

## Try a three-word opening

Run from the repository root with Python 3.12. Download the repository or clone it so the model checkpoints are present. This command can run on CPU using just the included weights:

```sh
python -m pip install -r requirements.txt
python -m script.stage_4_script.generate_joke --model lstm --prompt "What did the"
```

Type another three-word opening in place of `What did the`. For example, try `How do you` or `A man started`. Switch `--model` to `rnn` or `gru` to compare the same opening. No course data are needed for this command. Out-of-vocabulary words retain their spelling in the displayed prompt and share an unknown-word embedding inside the model.

The default loads the wider checkpoint and uses deterministic beam search (width 5, up to 32 continuation tokens). Its output for the first command is “What did the tin man say when he got run over by a steamroller? Curses! Foil again!” That complete joke occurs in the training split. The inference command generates each token from model weights; it never opens a corpus file. For more varied outputs, add `--strategy sample --seed 173`; to use the smaller 20-epoch model, add `--results-dir results/generation`.

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

## Readability and training overlap

For manual testing we also trained the three cells with wider embeddings (80), more hidden units (128), a lower learning rate (0.002), and 45 epochs. To reproduce, append `--embedding 80 --hidden 128 --learning-rate 0.002` to the training command, replace `--epochs 20` with `--epochs 45` and use a fresh output directory. Run the evaluation command on that directory. To verify the submitted wider checkpoints and their original top-k samples, append `--results-dir results/generation_wide` to the replay command.

In a small AI-assisted reading of shuffled beam outputs, the 20-epoch and 45-epoch models produced 9 and 10 readable main utterances, respectively, out of 12 each. A recognizable payoff appeared 4 and 8 times. Entire normalized training jokes reappeared 2 and 6 times. The four development openings were selected from validation text: two frequent training openings (`What do you`, `Did you hear`) and two whose exact three-token sequence had not appeared in training (`A man started`, `Did you see`). Each cell used all four. The rating file identifies the recipe after unblinding. This comparison informed the default demo choice; it is not an independent estimate of generalization. The wider models all had worse held-out perplexity (RNN 104.88, LSTM 60.60, GRU 109.34).

[Interactive examples](results/manual_testing/beam_outputs.csv) include all three course-style openings, all four development openings, and two extra openings. [Blind reading notes](results/manual_testing/blinded_comparison.csv) give the per-output judgments. Some unfamiliar prompts produce incomplete or unrelated sentences. The reproducible top-k samples produced during training use different decoding settings from the manual demo and are stored separately in each model folder.
