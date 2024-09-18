# Iteration 23. More external data

_12-09-2024_

## Goal

Is there more external data that can improve the accuracy of the model?

## Motivation

Using RE-ARC allowed to improve LB score from 10 to 16. Data is really important.

Recently I have noticed that [Simon Strandgaard](https://github.com/neoneye) ([kaggle](https://www.kaggle.com/neoneye)) is [creating data](https://github.com/neoneye/simon-arc-lab) for ARC challenge.
I want to explore that data and also search for additional datasets.

## Development

So far I have used RE-ARC, ConceptARC, 1D-ARC and MINI-ARC. Only RE-ARC and MINI-ARC showed significative improvements on evaluation metrics.

### Sources of information

- [Simon ARC lab (neoeye)](https://github.com/neoneye/simon-arc-lab), apparently host the code to generate datasets for ARC tasks.
  - [Neoeye tama dataset](https://github.com/neoneye/arc-dataset-tama)
  - [Simon datasets on huggingface](https://huggingface.co/neoneye)
  - [Dataset viewer](https://neoneye.github.io/arc/)
  - [Multiple datasets for ARC](https://github.com/neoneye/arc-dataset-collection/tree/main)
    - [ARC_synthetic_extend](https://github.com/frankaging/ARC_synthetic_extend) At a first glance it seems that they have only changed the colors, this is not useful because we can do it with data augmentation.
    - [IPARC](https://github.com/neoneye/arc-dataset-collection/tree/main/dataset/IPARC). Simon says it is very hard and I agree, I have checked some examples and were hard to understand.
    - [PQA: Perceptual Question Answering](https://github.com/neoneye/arc-dataset-collection/tree/main/dataset/PQA) This dataset looks interesting.
    - [ARC Community](https://github.com/neoneye/arc-dataset-collection/tree/main/dataset/arc-community) The tasks are hard to understand.
    - [ARC dataset diva](https://github.com/neoneye/arc-dataset-diva) The arc-dataset-diva is focused on tiny tasks, where 2 pixels goes in, some transformation happens, and 2 pixels comes out. Probably too small, like the 1D-ARC dataset.
    - [dbigham ARC tasks](https://github.com/neoneye/arc-dataset-collection/tree/main/dataset/dbigham), 21 tasks. some of them have uncertainty on the test sample.
    - [synth_riddles](https://github.com/arc-community/synth_riddles) I don't like them, I don't understand some of them.
- Small datasets available on Kaggle
  - [nosound's 9 hand crafted ARC tasks](https://www.kaggle.com/datasets/zaharch/arc-nosound-tasks)
  - [Andy Penrose's 5 tasks](https://www.kaggle.com/datasets/andypenrose/extra-arc-tasks-for-testing)
- [ARC Public resources Google Sheet](https://docs.google.com/spreadsheets/d/1fR4cgjY1kNKN_dxiidBQbyT6Gv7_Ko7daKOjlYojwTY/edit?gid=167693902#gid=167693902)
  - [Language-complete Abstraction and Reasoning Corpus (LARC)](https://github.com/samacqua/LARC) I could use this
    dataset to test if using language definition of the tasks is useful. A following step would be
    to use code.
  - [ARC gym](https://github.com/SimonOuellette35/ARC_gym): a data generation framework for the Abstraction & Reasoning Corpus

There is one weird thing, why simon does not have its own data on the viewer?

## Results

### Add more external datasets

| experiment          | accuracy | correct_pixels | correct_size | pass_32 | vote_2 |
|---------------------|----------|----------------|--------------|---------|--------|
| baseline            | 7.62%    | 70.89%         | 88.64%       | 23.25%  | 15.91% |
| add new datasets    | 7.86%    | 71.21%         | 88.74%       | 23.00%  | 15.66% |
| add neoneye tama    | 7.46%    | 71.33%         | 89.12%       | 22.50%  | 15.66% |
| add MINI-ARC        | 7.62%    | 71.41%         | 89.21%       | 26.62%  | 17.05% |
| remove neoneye tama | 7.38%    | 71.22%         | 88.86%       | 24.75%  | 16.92% |

The differences between experiments are small and probably not significative, let's make a brief summary of the added datasets:

- neoneye's tama: 50 tasks with 100 variations each
- PQA: 7 different tasks with lots of variations
- MINI-ARC: 149 tasks with 4.5 samples per task
- Kaggle: 14 tasks with 3.8 samples per task

### Pretrain on datasets with more samples per task

Some of the datasets such as RE-ARC, PQA or neoneye's tama have a lot of samples for each task. Thus it might
have sense to first pre-train the model on those datasets and on a second stage use all the available data.

| experiment                              | accuracy | correct_pixels | correct_size | pass_32 | vote_2 |
|-----------------------------------------|----------|----------------|--------------|---------|--------|
| pretrain on big datasets + normal train | 10.11%   | 72.79%         | 89.06%       | 28.25%  | 20.20% |
| double length train                     | 9.93%    | 71.73%         | 87.80%       | 26.88%  | 17.30% |

We see a small improvement, it might not be significative.

## Conclusion

We have tried adding new external datasets to train. Results are not conclusive, it is not clear if adding
this new datasets improves the validation scores.

## Next steps

- [ ] [Language-complete Abstraction and Reasoning Corpus (LARC)](https://github.com/samacqua/LARC)

## TODO

- [x] [PQA: Perceptual Question Answering](https://github.com/neoneye/arc-dataset-collection/tree/main/dataset/PQA)
  - [x] Read the paper
  - [x] The dataset is big, how to deal with it? Can I group all the common tasks together?
  - [x] Check the colors
- [x] Visualize Simon datasets https://github.com/neoneye/simon-arc-lab. I have been looking at the code, but I don't see how to decode the datasets. I believe he only works with RLE encoded data.
  - [x] https://github.com/neoneye/simon-arc-lab/blob/main/simon_arc_lab/rle/deserialize.py
- [x] Create a small dataset combining the 2 existing small kaggle datasets
- [ ] [ARC gym](https://github.com/SimonOuellette35/ARC_gym)
- [x] Could it have sense to pretrain only on the datasets that have a lot of variation like RE-ARC and PQA?
- [x] Does neoneye tama improve accuracy?
