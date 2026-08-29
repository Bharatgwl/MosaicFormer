---
language:
  - en
license: other
library_name: pytorch
tags:
  - pytorch
  - language-modeling
  - transformer
  - memory-compression
  - research-prototype
datasets:
  - roneneldan/TinyStories
---

# MosaicFormer

MosaicFormer is a small decoder-only language-model research prototype. It
modifies a GPT-style Transformer by combining:

- local causal attention for nearby token detail
- learned memory compression for older chunks
- memory cross-attention for retrieving compressed history
- gated fusion for mixing local context and memory context

This model is currently a research and learning prototype, not a production
language model.

## Intended Use

MosaicFormer is intended for:

- educational study of Transformer internals
- small-scale language-modeling experiments
- ablation studies around memory compression
- long-context recall experiments
- research report and paper preparation

It is not intended for:

- production deployment
- high-stakes decisions
- factual question answering
- medical, legal, financial, or safety-critical use

## Architecture

The model processes input sequences in chunks. Each chunk is handled with local
causal attention. After a chunk is processed, the model compresses it into a
small number of learned memory tokens. Later chunks can attend to these memory
tokens, and a learned gate decides how much to use local context versus memory
context.

For a beginner-friendly architecture explanation, see
`UNDERSTANDING_MOSAICFORMER.md`.

## Training Data

Experiments so far have used small local datasets and TinyStories-style text.
Any public release should clearly state the exact dataset files, dataset
versions, preprocessing steps, token counts, training epochs, and random seeds.

Known local experiment datasets:

- `the-verdict.txt`
- `tiny_shakespeare.txt`
- `tinystories_1000.txt`
- TinyStories-derived text files

## Evaluation

Current evaluation includes:

- validation loss
- perplexity
- training time
- peak GPU memory
- parameter count
- synthetic needle-recall accuracy

Example TinyStories-style result from local experiments:

| Model | Parameters (M) | Validation Loss | Perplexity |
| --- | ---: | ---: | ---: |
| Baseline GPT same width | 13.690 | 4.4135 | 82.56 |
| Baseline GPT parameter matched | 14.144 | 4.3845 | 80.20 |
| MosaicFormer local only | 13.690 | 4.3566 | 77.99 |
| MosaicFormer memory no gate | 14.089 | 4.3354 | 76.36 |
| MosaicFormer full | 14.221 | 4.3118 | 74.58 |

Example synthetic recall result:

| Model | Recall Accuracy |
| --- | ---: |
| Baseline GPT parameter matched | 1.0000 |
| MosaicFormer full | 0.9688 |

These are early local results. They should not be treated as final benchmark
claims until repeated across multiple random seeds and larger datasets.

## Limitations

- The current implementation is slower than the GPT baseline in local tests.
- The memory path improves TinyStories validation loss in one run, but more
  seeds are required.
- Synthetic recall performance is strong but does not yet beat the
  parameter-matched GPT baseline.
- The model is small and trained on limited data.
- Generated text may be repetitive, incorrect, or low quality.

## Ethical Considerations

This is a small experimental language model. It may reproduce biases or unsafe
patterns from training data. Do not use it for consequential decisions or as a
source of factual truth.

## Ownership And License

Copyright (c) 2026 Bharat Gusaiwal. All rights reserved.

This project is currently released under an All Rights Reserved notice unless a
different license is explicitly added later. Third-party datasets, libraries,
and dependencies remain governed by their own licenses.
