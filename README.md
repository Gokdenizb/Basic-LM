# Basic-LM

A word-level LSTM language model trained from scratch on WikiText-2. It starts from random weights and estimates `P(next word | previous words)`.

## Concepts

- A sentence probability is the product of next-word probabilities (the chain rule). An n-gram estimates each factor from counts. These models estimate it with a network.
- An embedding is a learned vector looked up from a word id.
- One LSTM step reads that vector and the previous hidden state, then writes a new hidden state. That state is the memory of the prefix.
- A linear layer produces a score for every vocabulary word. Softmax turns the scores into probabilities.
- Training loss is cross-entropy: the negative log probability of the true next word.
- During training the input is the real previous words. During generation the model appends its own picks.
- Perplexity is `exp(average cross-entropy)`. Lower is better.
- The vocabulary is built from the training split only. A validation or test word that was never kept becomes `<unk>`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check the training device, then fit `y = 2x` once so the PyTorch loop is familiar:

```bash
python exercises/check_device.py
python exercises/fit_line.py
```

On this Mac the device is `mps`.

## Data

WikiText-2 raw (`Salesforce/wikitext`, config `wikitext-2-raw-v1`) is already split on whitespace, and it writes some punctuation as `@-@`, `@,@`, and `@.@`. The tokenizer turns those placeholders back into `-`, `,`, and `.`, drops blank lines and `= headings =`, and appends `<eos>` at the end of each line. Words that appear fewer than 3 times in training become `<unk>`. Case is kept.

```bash
python -m src.data
```

That prints the vocabulary size (32,993), token counts, the 20 most common training words, and one batch. Inputs and targets both have shape `(35, 20)`. The target is the input shifted forward by one token.

## Train

```bash
python -m src.train
```

The network uses embedding size 200, hidden size 200, 2 layers, and dropout 0.2. Training uses sequence length 35, batch size 20, AdamW at `1e-3`, gradient clip `1.0`, and 8 epochs. The checkpoint with the best validation perplexity is written to `checkpoints/lstm.pt`.

## Perplexity

Add-1 bigram smoothing is the count-based yardstick. Its denominator adds the full vocabulary size, so the number is high. The LSTM lands well below it.

| Model | Validation | Test |
| --- | ---: | ---: |
| LSTM | 221.53 | 200.76 |
| Add-1 bigram | 2227.78 | 2079.63 |

Validation perplexity fell from 371.13 after epoch 1 to 221.53 after epoch 8.

```bash
python -m src.eval --checkpoint checkpoints/lstm.pt
```

## Samples

Prompt: `The history of science`. 30 words.

Temperature 0 (always the most likely word) collapses. The top next word is `-` (about 9.7%), and a hyphen is often followed by a rare word that was mapped to `<unk>`:

```
- <unk>, <unk>, <unk>, <unk>, <unk>, <unk>, <unk>, <unk>, <unk>, <unk>, <unk>, <unk>, <unk>, <unk>, <unk>
```

Temperature 0.8:

```
in the first-time South Africa, and Love (and the <unk> <unk>). The Viscount's compositions has been also made to both the Augustan History
```

Temperature 1.2:

```
Love, Stokes retired on the Rock / Angel Songs in a original parliament! In 1874 that he aired on September 1. The second role was announced Rue
```

The same prompt changes the next-word list. After "The history of science" the top of the list is `-`, `,`, `:`, `and`, `fiction`. After a bare `The`, the mass sits on `<unk>` and words that start an article (`One`, `The`, `New`).

Local grammar and a Wikipedia tone are the realistic bar. The model does not write a coherent article.

```bash
python -m src.generate --checkpoint checkpoints/lstm.pt --prompt "The history of science" --words 30 --temperature 0.8
```

## Gradio

```bash
python app.py
```

Open http://127.0.0.1:7860. Type a starting phrase, set how many words to generate (10–100) and a temperature (0 means greedy), then press Generate. The table is the distribution over the next word for the phrase as typed, before any new words are appended.
