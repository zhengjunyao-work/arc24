# Data Understanding

## Collect initial data

<https://github.com/fchollet/ARC-AGI>

There are 400 training tasks and 400 evaluation tasks. The evaluation tasks are said to be more difficult
than the training tasks.

The test set is hidden and it has 100 new and unseen tasks.

The tasks are stored in JSON format.

## External data

<!--- It is allowed in this challenge? If so write it here ideas of how to find
it and if people have already posted it on the forum describe it. --->

There are some variations of the ARC dataset:

- [ConceptARC](https://github.com/victorvikram/ConceptARC) is a new, publicly available benchmark in the ARC domain that systematically assesses abstraction and generalization abilities on many basic spatial and semantic concepts. It differs from the original ARC dataset in that it is specifically organized around "concept groups" -- sets of problems that focus on specific concepts and that vary in complexity and level of abstraction. It seems to be **easier** than the original ARC benchmark.
- [Mini-ARC](https://github.com/ksb21ST/Mini-ARC) a 5 × 5 compact version of the ARC, was generated manually to maintain the original’s level of difficulty.
- [1D-ARC](https://github.com/khalil-research/1D-ARC) A simpler version of ARC tasks with only one dimension.
- [Sort-of-ARC](https://openreview.net/forum?id=rCzfIruU5x5) shares ARC’s input space but presents simpler problems
with 20×20 images containing three distinct 3×3 objects. I only could find the paper, not the dataset.
- [https://github.com/michaelhodel/re-arc](https://github.com/michaelhodel/re-arc) RE-ARC: Reverse-Engineering the Abstraction and Reasoning Corpus by Michael Hodel, member of MindsAI team
- [MC-LARC](https://github.com/MC-LARC/MC-LARC.github.io) Text descriptions for the ARC training set.
- [arc-generative-DSL-infinite-data](https://github.com/arc-community/arc-generative-DSL-infinite-data) Jack Cole repo "Slowly building a collection of infinite riddle generators for benchmarking data-hungry methods"
- [Abstract Reasoning Challenge - community resources](https://github.com/arc-community)
- [arc-dataset-collection](https://github.com/neoneye/arc-dataset-collection) Multiple datasets for ARC (Abstraction and Reasoning Corpus)

## Describe data

<!---Describe the data that has been acquired, including the format of the data,
the quantity of data (for example, the number of records and fields in each table),
the identities of the fields, and any other surface features which have been
discovered. Evaluate whether the data acquired satisfies the relevant requirements. --->

## Explore data

<!---This task addresses data mining questions using querying, visualization,
and reporting techniques. These include distribution of key attributes (for example,
the target attribute of a prediction task) relationships between pairs or small
numbers of attributes, results of simple aggregations, properties of significant
sub-populations, and simple statistical analyses.

Some techniques:
* Features and their importance
* Clustering
* Train/test data distribution
* Intuitions about the data
--->

## Verify data quality

<!---Examine the quality of the data, addressing questions such as: Is the data
complete (does it cover all the cases required)? Is it correct, or does it contain
errors and, if there are errors, how common are they? Are there missing values in
the data? If so, how are they represented, where do they occur, and how common are they? --->

## Amount of data

<!---
How big is the train dataset? How compared to the test set?
Is enough for DL?
--->
