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

## Results

## Conclusion

## Next steps

## TODO

- [ ]
