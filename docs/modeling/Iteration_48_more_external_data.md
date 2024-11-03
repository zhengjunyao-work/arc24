# Iteration 48. More External data

_03-11-2024_

## Goal

Can we improve the LB score by using more external data?

## Motivation

The paper [Combining Induction and Transduction for Abstract Reasoning](https://www.cs.cornell.edu/~ellisk/documents/arc_induction_vs_transduction.pdf)
along with a [400k tasks dataset](https://huggingface.co/collections/barc0/synthetic-arc-dataset-6725aa6031376d3bacc34f76) has
been just published.

It is very likely that training my models in this extra data could result on improved accuracy, so I have
to do it and do it fast.

## Development

### Download datasets

```
git lfs install
git clone git@hf.co:datasets/barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems
git clone git@hf.co:datasets/barc0/100k-gpt4-description-gpt4omini-code_generated_problems
git clone git@hf.co:datasets/barc0/100k-gpt4omini-description-gpt4omini-code_generated_problems
```

## Results

## Conclusion

## Next steps

## TODO

- [ ] Explore the dataset
- [ ] How can I train on this new dataset? It is much bigger than the other datasets
- [ ] Does it improve the evaluation accuracy?
- [ ] Does it improve the LB score?
- [ ] Train a model to generate code