# LinkedIn Post Draft: MosaicFormer

## Main Post

Over the last few weeks, I have been studying large language models from scratch and building a small research prototype called **MosaicFormer**.

The idea is simple:

Instead of letting every token attend to every previous token directly, MosaicFormer tries to divide context into two parts:

- **local attention** for nearby tokens, where exact detail matters
- **compressed memory** for older tokens, where the model keeps a smaller summary of past information

Then a **gated fusion** layer learns how much the model should trust the local context versus the compressed memory.

In simple words:

The model does not try to remember the whole past word-for-word. It keeps the nearby text clearly, and stores older text in a smaller memory form.

I compared it against a parameter-matched GPT baseline, so the comparison is not just "bigger model wins." Both models are around 14M parameters.

Some early WikiText-103 results:

```text
At around 60k training steps:
GPT baseline validation loss: 4.6082
MosaicFormer validation loss: 4.5874
```

I also added a synthetic needle-recall test to check whether the model can recover information placed earlier in the context. In that controlled test, both the baseline and MosaicFormer reached strong recall accuracy after enough training.

What I learned:

- The idea is valid as a student research prototype.
- MosaicFormer can learn and can slightly improve validation loss in early experiments.
- It is currently slower than the GPT baseline, so I cannot claim it is more efficient yet.
- The next important step is better optimization, longer fair training, harder recall tests, and repeated runs with multiple seeds.

This project taught me a lot about:

- transformer internals
- causal masking
- attention complexity
- memory compression
- fair baselines
- checkpoint training on Colab
- why research claims need careful experiments

I am keeping the claims modest for now:

**MosaicFormer is not a proven new state-of-the-art model. It is an early research prototype that explores compressed memory inside a small transformer.**

For now, I am keeping the full code, checkpoints, and raw experiment artifacts private while I finish the report and verify the results. The project is copyright protected and currently in a pre-publication stage. When the experiments are cleaner, I plan to share a proper technical write-up and a model card with clear limitations.

Next, I plan to:

- finish longer checkpoint-based training
- compare at equal training steps
- add training curves
- run harder memory-recall experiments
- prepare a technical report
- release a clean model card when the results are ready

This started as a learning project, but it is slowly becoming my first serious research-style experiment.

#LLM #DeepLearning #Transformers #PyTorch #MachineLearning #AIResearch #StudentResearch

## Shorter Version

I have been studying LLMs from scratch and building a small research prototype called **MosaicFormer**.

The idea: combine local attention with compressed memory, then use gated fusion to decide how much the model should rely on nearby context versus older summarized context.

I compared it with a parameter-matched GPT baseline, both around 14M parameters.

Early WikiText-103 partial result:

```text
At around 60k training steps:
GPT baseline validation loss: 4.6082
MosaicFormer validation loss: 4.5874
```

Important: I am not claiming this is state of the art. MosaicFormer is currently slower than the baseline, and it needs longer fair training, harder recall tests, and repeated runs.

The full code and checkpoints are private for now while I finish verification and prepare a proper report.

But as a student research prototype, it has been a very useful experiment. I learned a lot about transformer internals, memory compression, causal masking, fair baselines, checkpointing, and honest research reporting.

Next step: finish equal-step training, generate curves, and write a proper technical report.

#LLM #Transformers #PyTorch #DeepLearning #MachineLearning #AIResearch

## Comment To Add Under The Post

One lesson from this project: getting a model to train is not the same as proving a research idea.

The hard part is making the comparison fair:

- same parameter scale
- same data
- same training steps
- same tokenizer
- same validation method
- no future-token leakage
- repeated runs

That is the part I am working on next.

## Private Note Before Posting

Before posting publicly:

- Do not attach the full source code if you still want to keep the project private.
- Do not upload checkpoint files in the LinkedIn post.
- Use the architecture diagram and selected validation-loss numbers only.
- Say "early prototype" or "student research prototype", not "state of the art".
- Say "partial WikiText-103 run" unless both models have completed equal-step training.
- Keep the GitHub repository private until the report, license, and model card are ready.
