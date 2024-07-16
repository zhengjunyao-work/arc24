# Modeling

## Space of possible solutions

To solve the ARC challenge we need to do two things:

1. Understand what the transformation is. This is the abstract reasoning step. We need to build a representation of the input to be able to do that:
   - Deep learning model
     - LLM
     - Vision model
   - Python code. The grids are not as complex as real images, so we could build python code to extract
     representations from the grids.
2. Implement the transformation. This could be done:
   - With code
     - Using python code
       - With primitive functions
       - Without primitive functions
     - With a Domain Specific Language (DSL)
   - Using a model (very likely an LLM)

### Icecuber approach: DSL

It is possible to solve the challenge given enough compute and a complete DSL. Validating a solution
is cheap and fast, this approach will try a huge number of solutions until it finds the correct one.

It skips completely the abstraction and reasoning and uses brute force to search the solution.

Clearly this is not the path to more intelligent systems but we must notice that this approach could
work and will scale well with compute.

### MindsAI team approach

Little details are given about their approach but in a nutshell is something like this:

1. Fine-tune an LLM on augmented ARC tasks. Probably on the re-arc dataset, maybe a bigger version of it.
2. On inference they augment the test samples (I don't know how) and fine-tune the LLM again on those tasks
3. Make predictions with the LLM

By fine-tuning an LLM on ARC tasks the LLM will likely learn a good representation of the grid. However
they have said that without test time fine-tuning they will only score 1% so that step is crucial (they are currently at 39%)

### Ryan Greenblatt approach

Ryan uses GPT4o so he cannot fine-tune the model. He has to rely on his default knowledge. In the other hand it can
make use of the strong programming abilities and revision capabilities of the top models.

He uses few-shot prompt with meticulous step-by-step reasoning and handcrafted better grid representations.

### Single model LLM

Given the inputs the LLM will reason with text what the transformation is and apply it to the test sample.

This model is capable of generating an output given:

- Task text description and input
- Input-output pairs and input
- Task text description, input-output pairs and input

Thus it is a pretty general model, the previous capabilities will be used at evaluation, but in addition to them
the model is also capable of:

- Describe the grids and answer questions about them (showcasing that it has learned a good representation of the grids)
- Describe changes and constants between pairs of grids
- Generate more inputs from some distribution
- Guess if some grid belongs to a distribution of grids
- Predict which symmetries could be applied to some task without changing the meaning (similar to the previous point)

To be able to do abstraction from a few examples a good representation is needed.

That same representation could be used by the LLM to generate the output.

On evaluation the goodness of the text description could be validated against the examples, leaving one out.
This could enable revisions and refining of the solution.

This approach could be enhanced with test time fine-tuning that has been probed to work.

Training this model will require a grid generator that also outputs text descriptions. It could
be later replicated with 3d renderings in the real world.

```
text -> image | easy
image -> text | hard
```

This is true for real world data, it is also true for ARC? I'm not sure but we will find out. When
we build the grid generator we will see how easy is to create different representations using python code.

This is arguably the most similar model to a human person.
This approach is similar to MindsAI team (little details are known). The strength of this approach is
that learning all those tasks together will hopefully create a strong and general model.

### LLM to generate python code

Instead of doing the transformation directly with the LLM, we ask the LLM to write python code that
does the transformation. In fact it could be the same model that could be able to do both tasks!

Writing python code to do the transformation is non trivial, many times we have to deal with
objects and that is not easy.

Having a set of primitive functions could make the problem easier. Also being able to work with high
level abstractions or representation of the grids could also simplify the problem.

The goodness of this approach is that we can test the python programs easily using the examples.

It requires a base model that is good at coding. Fine-tuning is more difficult because we need good
python solutions. Maybe an iterative approach could work:

- Solve n problems
- Fine-tune on those solutions
- Solve additional m problems
- Fine-tune on those solutions
- Repeat...

### DSL + Intuition

This requires a complete DSL which is not trivial to build.

The intuition module will give importance to each of the DSL functions given the examples

To work well the intuition needs to build a good representation of the examples. Abstraction is all about learning a good representation.

### Few-shot prompting

A really intelligent LLM could solve the ARC challenge using few-shot prompting. The prior knowledge
would be injected in those few-shot samples. It is very likely that given text descriptions and step-by-step
reasoning of the tasks would be helpful.

This approach would require a big context window because tokenizing the examples can require a considerable
window size.

## Select modeling technique

<!---Document the actual modeling technique that is to be used. If multiple
techniques are applied, perform this task separately for each technique.
Many modeling techniques make specific assumptions about the data—for example,
that all attributes have uniform distributions, no missing values allowed,
class attribute must be symbolic, etc. Record any such assumptions made. --->

## Generate experimentation design

<!---Describe the intended plan for training, testing, and evaluating the models.
A primary component of the plan is determining how to divide the available dataset
into training, test, and validation datasets.

Doing a plot of score vs train size could be helpful to decide the validation strategy

Depending on the size of the data we have to decide how we are going to use submissions.
The less the submissions the most confidence we can have on the score. However sometimes
the data distribution is very different, or the size of the data is small and we have
to make a lot of submissions. Sometimes is not easy to have a good correlation between
validation score and LB score
--->
