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

## Results

## Conclusion

## Next steps

## TODO

- [ ] Have a look at RE-ARC, maybe I can reuse it.
- [ ] How difficult is to be data augmentation resistant? If it is difficult I should better concentrate on creating new tasks.
