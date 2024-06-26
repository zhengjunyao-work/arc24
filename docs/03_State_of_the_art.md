# State of the art

<!--- --->

## Papers

These are the sources of papers used:

- [Citations to the "On the measure of intelligence" paper on Google Scholar](https://scholar.google.com/scholar?start=10&hl=en&scisbd=1&as_sdt=2005&sciodt=0,5&cites=645844335140263496&scipsc=)
- TODO: [Papers on Arxiv with `abstraction reasoning corpus` in the title](https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=abstraction+reasoning+corpus&terms-0-field=title&classification-physics_archives=all&classification-include_cross_list=include&date-filter_by=all_dates&date-year=&date-from_date=&date-to_date=&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first)

### ⭐ [Neural networks for abstraction and reasoning: Towards broad generalization in machines](https://arxiv.org/abs/2402.03507)

Nicely written paper that tries to solve the ARC challenge with two methods:

1. Dreamcoder. Is a method to create programs given a set of primitive functions
2. LLMs. They show that using transpose and rot90 augmentations can double the accuracy of the models. This highlights the sequential and non-2d nature of the typical LLM data.

Results are weak and do not surpass the state of the art.

> Abstraction and reasoning - developing computer systems that can learn new concepts from a small number of examples, something that humans find relatively easy

<!--- --->

> We revisit whether new advances can allow computers to extrapolate to new concepts rather than merely interpolate.

<!--- --->

> With only a handful of training examples per ARC task, and 10900 possible answers (of which exactly one gains credit), traditional machine learning (ML) methods that require large datasets have so far been unable to make progress.

<!--- --->

> Analogy-making has been considered central to the notion of intelligence. When presented with novel situations (for example; opening a new type of door, or conversing about a new topic), humans effortlessly solve these situations by creating analogies to previous experiences and concepts.

<!--- --->

> Chollet notes that while excellent progress has been made in solving specific tasks to approach or surpass human-level (such as detecting cats and playing Go), these models generally require a huge amount of training and are limited to performing well on situations that they were trained on. The failure of neural network models to perform when extrapolating outside the training data has been widely explored

<!--- --->

> Inductive programming describes algorithms that derive programs that explain a series of examples.

<!--- --->

> when a person attempts an ARC task, we see a similar process: one tries to abstract the
training tasks to a ‘program’ (generally in natural language in one’s head, for example “rotate by 90
degrees”); human intuition is the search procedure.

### ⭐ [LLMs and the Abstraction and Reasoning Corpus: Successes, Failures, and the Importance of Object-based Representations](https://arxiv.org/abs/2305.18354v2)

This paper highlights the fact that LLMs have difficulties understanding the 2d nature of the ARC dataset.

1. Evaluate the model with few-shot prompt as it is my idea, giving reasoning as input. However I don't believe this is done as extensively as I want to do myself
2. Create a 1D ARC dataset where the models perform better
3. Using a object based representation of the problem where GPT4 solves 23/50 problems (an easy subset from ARC)

What would be the best input for the LLMs to understand 2d structures?
I could fine-tune a model to do row, col, diagonal addition or presence detection. This might force the model to create a great representation for the problem.

> Reasoning is using evidence, arguments, and logic to arrive at conclusions or make judgments

<!--- --->

> A closer examination of the tasks that GPT-4 solved correctly using the direct-grid approach reveals some interesting patterns in the reasoning provided by the model. Out of the 13 tasks that were correctly solved, only three tasks were accompanied by the correct reasoning steps.

<!--- --->

> To address the challenges we have identified thus far and to enhance LLM performance, we propose the integration of an external tool to aid in producing an object representation of a task. More specifically, we leverage the ARGA algorithm to execute object abstraction before prompting LLMs for the solution.

<!--- --->

> Our exploration started with a straightforward, grid-based textual encoding approach, which revealed that GPT struggles due to the non-sequential representation of complex objects in text. We then introduced the 1D-ARC, a simplified, single-dimensional version of the ARC. By reducing the task complexity and dimensionality, we aimed to make ARC tasks more approachable for LLMs. Our evaluations on the 1D-ARC indicated improvements in performance but also highlighted that simplification alone could not bridge all the gaps in GPT’s reasoning processes. In the third phase of our exploration, we adopted an object-based approach, integrating an external tool, the ARGA framework, to assist in object abstraction. This led to significant improvements in GPT’s problem-solving abilities, reaffirming the importance of structured, object-based representations in complex reasoning tasks.

### ⭐ [Addressing the Abstraction and Reasoning Corpus via Procedural Example Generation](https://arxiv.org/abs/2404.07353)

This paper presents the re-arc repo, which allows to generate at least 10k new samples for each task in the ARC training dataset.

Could I modify it to output text descriptions of the synthetic inputs? That could allow the model to learn a good representation of the grids and also to learn what the transformation is.

> The sample-efficiency of learning algorithms might be im- proved by building a curriculum that increases example difficulty over the course of training - as opposed to training on instances of the full range of difficulties throughout the entire training

<!--- --->

> Each generator is a standalone Python function merely making use of the DSL and functions from the random module from the standard library. The median generator consists of 40 lines of code and uses 22 DSL primitive calls and 10 calls to the random module

It is curious that so many (22) primitive function calls are needed on median.

My rough estimation of primitive functions in [arc-dsl](https://github.com/michaelhodel/arc-dsl) is 160 (count the number of occurrences of `def `). We know that this set of primitives is complete for the train set, but is it for the evaluation and test set?

### [Large Language Models Are Not Strong Abstract Reasoners](https://arxiv.org/abs/2305.19555)

Results on ARC challenge are very weak, but they don't add task descriptions that I believe would have helped the models (in addition to few-shot prompting)

> Abstract reasoning is a fundamental task for cognition, consisting of finding and applying a general pattern from few data

<!--- --->

> We perform extensive evaluations of state-of-the-art LLMs, showing that they currently achieve very limited performance in contrast with other natural language tasks, even when applying techniques that have been shown to improve performance on other NLP tasks.

<!--- --->

> In this paper, we present what is, to the best of our knowledge, the first extensive evaluation of Large Language Models for abstract reasoning. We show that LLMs do not perform well on all types of tasks, although not all models are equally poor. Prompting and refinement techniques that improve performance on NLP tasks do not work for abstract reasoning. Our experiments show that the bottleneck in the performance lies in the recognition of new unseen abstract patterns and not in a lack of understanding of the task or the prompt. These results hold in discriminative settings, where the models must find the correct answer within a small set of propositions. A qualitative study of selected failure cases in the appendix further reveals that models tend to reason inconsistently and in a shallow way. We hypothesize that current self-supervised autoregressive LLMs lack fundamental properties for strong abstract reasoning tasks and human-like cognition. We posit that methods based on causal reasoning and program induction could help improve the reasoning abilities of neural networks.

### [Comparing Humans, GPT-4, and GPT-4V On Abstraction and Reasoning Tasks](https://arxiv.org/abs/2311.09247)

This paper shows an evaluation of GPT4 on ConceptARC dataset, an easier version of ARC dataset that has well defined categories.

However there isn't task description so that is equivalent to trying to learn MATH problems with just the answer and not the reasoning.

> The defining characteristic of abstract reasoning is the ability to induce a rule or pattern from limited data or experience and to apply this rule or pattern to new, unseen situations.

<!--- --->

> Here, we performed evaluations using a more informative, one-shot prompt for text versions of tasks, and experimented with similar zero- and one-shot prompts for the multimodal case in which task-grids were given as images. We found that our more informative one-shot prompt improved GPT-4’s performance in the text case, but its performance remained well below that of humans and the special-purpose Kaggle-ARC program.

### [Large Language Models as General Pattern Machines](https://arxiv.org/abs/2307.04721)

This seems to be one of the initial evaluations of ARC using LLMs. The results are modest, but they show they can solve the problems for arbitrary symbols (not just 0-9 but any symbol)

> The capacity of LLMs to act as general pattern machines is driven by their ability to perform in-context learning on sequences of numeric or arbitrary tokens.

<!--- --->

> Observation: consistent tokenization matters.

<!--- --->

> Observation: token mapping invariance. The hypothesis that LLMs can serve as general pattern machines stems from the observation that they can still solve a non-trivial number of ARC problems using alphabets A sampled randomly from the LLM’s token vocabulary.

### [Teaching Large Language Models to Reason with Reinforcement Learning](https://arxiv.org/abs/2403.04642)

### [Reasoning Abilities of Large Language Models: In-Depth Analysis on the Abstraction and Reasoning Corpus](https://arxiv.org/abs/2403.11793)

### [Can Large Language Models Learn Independent Causal Mechanisms?](https://arxiv.org/abs/2402.02636)

### [Learn Abstraction in an Abstract Way: The Long Journey Ahead](https://openreview.net/forum?id=wHanWNJN0r)

### [Do Large Language Models Solve ARC Visual Analogies Like People Do?](https://arxiv.org/abs/2403.09734)

## Repos

- [arc-dsl](https://github.com/michaelhodel/arc-dsl) Domain Specific Language for the Abstraction and Reasoning Corpus by Michael Hodel, member of MindsAI team
- [https://github.com/michaelhodel/re-arc](https://github.com/michaelhodel/re-arc) RE-ARC: Reverse-Engineering the Abstraction and Reasoning Corpus by Michael Hodel, member of MindsAI team

## Videos

I could use [downsub](https://downsub.com/) to get subtitles from a Youtube video.

### ⭐ [Dwarkesh Patel | Francois Chollet - LLMs won’t lead to AGI - $1,000,000 Prize to find true solution](https://www.youtube.com/watch?v=UakqL6Pj9xo)

### ⭐ [Machine Learning Street Talk | Chollet's ARC Challenge + Current Winners](https://youtu.be/jSAT_RuJ_Cg?si=-s_XpeeDA2BQYlVy)

> Basically that we use so many training examples.
> It's it's not to necessarily teach it so many concepts.
> It's to teach it a space around the concepts and to also prevent it
> from kind of using the shortcuts that the models are prone to.

<!--- --->

> Ideally we would train a model on internet data and generalize to the ARC dataset, that's what Chollet would love to see.

<!--- --->

> So I just figured, okay, how well could we do?
> Uh, even just something very simple, like learning a task in isolation.
> If we had an unlimited number of examples for a given task.

<!--- --->

> Probably 20 different kind of many experiments in formatting the data in various ways.

<!--- --->

> If you train a model on the re-arc dataset you will get like 1% on the test set. But if you apply their
> techniques of active inference the score will increase to 23%

<!--- --->

> There are some scaling laws that suggest that the bigger the model the less test data needs to learn

<!--- --->

> The DSL has 160 functions, but the author believes it could rewrite it to be just 30

Their method is:

1. Fine-tune an LLM on augmented ARC tasks. Probably on the re-arc dataset, maybe a bigger version of it.
2. On inference they augment the test samples (I don't know how) and fine-tune the LLM again on those tasks
3. Make predictions with the LLM

### [LLMs as a system to solve the Abstraction and Reasoning Corpus (ARC) Challenge!](https://www.youtube.com/watch?v=plVRxP8hQHY)

## Conclusions

### Definitions of abstraction and reasoning

> Abstract reasoning is a fundamental task for cognition, consisting of finding and applying a general pattern from few data

<!--- --->

> The defining characteristic of abstract reasoning is the ability to induce a rule or pattern from limited data or experience and to apply this rule or pattern to new, unseen situations.

<!--- --->

> Abstraction and reasoning - developing computer systems that can learn new concepts from a small number of examples, something that humans find relatively easy

> Reasoning is a knowledge acquisition efficiency

## TODO

- [ ] Jack Cole approach
- [ ] Buck approach
- [ ] Icecube approach
- [ ] What is the best way to encode 2d information for an LLM like Llama3?
- [ ] How can we learn from few examples? Do we need a good representation of the data? Why ML methods need huge datasets? That is where the priors kick in, those priors influence the representation of the data.
- [ ] Search more relevant papers
  - [ ] https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=abstraction+reasoning+corpus&terms-0-field=title&classification-physics_archives=all&classification-include_cross_list=include&date-filter_by=all_dates&date-year=&date-from_date=&date-to_date=&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first
  - [ ] 
  - [ ] Read Kaggle's forum to see if more relevant papers were added
- [ ] Contrastive learning. Can a model predict if two input/output pairs belong to the same task?
