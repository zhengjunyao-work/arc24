# Iteration 29. Qwen 2.5

_25-09-2024_

## Goal

A new release of Qwen was announced yesterday: Qwen 2.5, does it improve the accuracy on ARC?

## Motivation

Simply swapping the model might bring improvements for free!

## Development

- https://qwenlm.github.io/blog/qwen2.5-llm/
- https://qwenlm.github.io/blog/qwen2.5-llm/#qwen25-05b15b3b-performance
- https://qwenlm.github.io/blog/qwen2.5-llm/#qwen25-05b15b-instruct-performance

> The size of the pre-training dataset is expanded from 7 trillion tokens to a maximum of 18 trillion tokens.

Whe looking at benchmarks we see a noticeable improvement between Qwen 2 and 2.5.

## Results

### Validation results when training different models for 10k steps

| model                 | accuracy  | pass_32    | vote_2     |
|-----------------------|-----------|------------|------------|
| Qwen2-0.5B            | 8.24%     | 26.50%     | 15.91%     |
| Qwen2-0.5B-Instruct   | 8.25%     | **26.75%** | 15.91%     |
| Qwen2.5-0.5B          | **9.37%** | **26.75%** | **18.31%** |
| Qwen2.5-0.5B-Instruct | 8.98%     | 26.00%     | 17.93%     |

Both versions of `Qwen2.5` achieve better results than `Qwen2-0.5B-Instruct`.

### Why models that aren't instruct are slower?

Inference with Qwen2.5B took 4920.6132 seconds, compared to the typical 1809.6193.  Why?

Inspecting the responses I have found that the non-instruct versions do repeat the prediction multiple times.
E.g.

    <|im_start|>assistant
    ### Output

    ```grid shape: 6x6
    1 595959
    2 181818
    3 959595
    4 818181
    5 595959
    6 181818
    ```
    Assistant
    ### Output

    ```grid shape: 6x6
    1 595959
    2 181818
    3 959595
    4 818181
    5 595959
    6 181818
    ```
    Assistant
    ...

## Conclusion

We have observed improvements when replacing Qwen2 by Qwen2.5. The most promising model is the non-instruct
version but there is a problem at inference: it does not stop predicting. Until that problem is not
solved I will use Qwen2.5

## Next steps

## TODO

- [x] Do the same experiment just changing the base model and compare the validation results
- [ ] Could I use Qwen-2.5-0.5B and avoid having repetitions in the prediction?
  - [ ] Check the training data.
  - [ ] Check the tokenizer
  - [ ] Local experiments
  - [ ] Maybe adding some new stopword to VLLM could be a quick fix
