# Iteration 26. Dawn of Omni-ARC

_16-09-2024_

## Goal

- Create a new dataset with python code that solves the ARC training tasks.
- Create a set of primitive functions that are used to solve many tasks (DSL)
- Create new tasks inspired by the training tasks and using the primitive functions
- Learn more about the ARC tasks

## Motivation

To learn a good representation of the data I believe that I should take the Omni-ARC approach: learn
many tasks related to ARC. The most promising task would be to write code that implements the ARC tasks.
It will allow to verify if it runs on the training samples giving a great advantage over voting.

This is going to be a very experimental iteration, thus I have many goals because I'm not sure what
will be the optimal approach to do this.

## Development

- The code should be able to cope with data augmentation, and I should test that
- I'm going to create new modules to store the code. The library `inspect` is useful to retrieve the source code.

### Thoughts when implementing the tasks

- I need to implement an object class with properties such as size, position and methods such as is_line, is_rectangle, move...
- Sometimes we need to ignore the colors when dealing with objects, other times we have to
  consider that different colors implies different objects
- After visualizing 50 tasks it seems that I can write easily code for 30 of them, so 60%
- There are many growing patterns that are probably better defined by example than by code.
- Many tasks involve grids
- If I make the assumption that grids are numpy arrays code will be simpler, I would simply have to write a wrapper around the tasks to be convert to list again
- Applying data augmentation to the source code might be more difficult than I believe. For example on `task_007bbfb7` there are some axis used that should not change. However `task_0520fde` uses axis and should be changed if there are rotations. Maybe I can solve it simply by adding some special comments that will be removed afterwards.
- Some implementations like `task_05269061` are hard to modify for geometric augmentation

### Specification

I want to write a python library that generates Abstraction and Reasoning Corpus (ARC) tasks. A task
will have some input grids, the python code that does the transformation of the task and some output grids.
Given this data a model could learn to generate the code given the inputs and the outputs.

This are the main ideas behind the library:

- **Focus on running code**. At first I thought about writing pseudocode and using non-implemented functions.
  But now I realize that the greatest value from this approach is being able solve the tasks using python
  code. Thus I need running code. Moreover having running code enables to do automated testing, and
  we don't have to deal with data augmentation issues. The data augmentation will be applied just
  to the inputs, not to the function.
- **As many different tasks as possible**. Experiments suggest that is better to have many different tasks
  than to have many samples from the same task.
- **High quality tested code**. Having clean code with the right level of abstraction will be crucial
  to be agile and speedup the writing of new tasks. Using automated tests will enable refactoring and
  agile development.
- **Small but flexible domain specific language**. This is much easier to maintain and expand that having
  a ton of specific functions. Moreover it will be easier to learn for the model if the number of
  primitive functions is small.

#### Task components

A task will always have the following components:

- Input grids
- Output grids
- Code

There are infinite ways to implement the task generator, but the output should follow the definition above.

The tasks are implemented using python dict. The grids are lists and the code will simply be a string.

#### Implementation requirements

The input grids for a task could come from an already created dataset, or there could be a function
that generates new input grids. When defining/implementing a new task we should specify what the
input is.

The input grids might undergo some data augmentation until they become the final inputs for the task.
This should also be specified when defining the task. This is more likely to happen when we are not
using a generator function, if the inputs come from a dataset using data augmentation will increase
the variability of the samples.

There might be some python function that implements the core task, but we can create new tasks by composing
multiple tasks. This should be specified when defining the task, some kind of pipeline of functions.
But to train the model we need to have all the code combined in a single function. So we need some
method to take a pipeline of functions and create a single function with all the code from those functions.
Ideally we could create multiple tasks very fast by specifying some core task and listing all the possible
tasks that could be used to compose new tasks.

There would be some task register where I store all the created tasks, that can be used later
to generate training data.

Internally I will work with numpy arrays, that makes the code easier. A wrapper will convert the list
to array at start and viceversa at the end.

### New repo

I have decided that I have to create a new repo to host all the code to generate tasks with code. A
good name for the repo would be omni-arc, since it will enable the training of the omni-arc models.

## Results

## Conclusion

## Next steps

## TODO

- [ ] Have a look at RE-ARC, maybe I can reuse it.
- [ ] How difficult is to be data augmentation resistant? If it is difficult I should better concentrate on creating new tasks.
