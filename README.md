# mlcap - Imperial College Machine Learning Professional Certificate Capstone Project

This repository chronicles my work on the "Black Box Optimisation Challenge" which is part of the machine learning course I am taking at Imperial College.  

## Background
We are provided a few data points from eight functions.  These range from 2 inputs --> 1 output to 8 inputs and one output.  Our task is to maximise the output of the function through a series of measurements, which we are allowed to make once a week for about 13 weeks.   Real-world scenarios that look like this project include problems where getting additional data is expensive and time-consuming, and where improving output has significant economic impact:

     - Hyperparameter tuning for Neural Network training (number of hidden layers, nonlinearities, number of hidden neutrons) 

     - Manufacturing process tuning, where a series of parameters are varied and we attempt to determine if the factory output increases or decreases

    - Exploration, for example for oil or minerals

## Function Inputs and outputs

Functions are as follows:

f_1():  R2 --> scalar output
f_2():  R2 --> scalar output
f_3():  R3 --> scalar
f_4():  R4 --> scalar
f_5():  R4 --> scalar
f_6():  R5 --> scalar
f_7():  R6 --> scalar
f_8():  R8 --> scalar

The input space is limited to the interval [0,1] along each dimension.  Each week, we submit one query in the following format (example for a 3-d function):

         0.000000-1.000000-0.775510

We then receive a series of eight scalar outputs, one per function, in a file containing, for example,

[np.float64(5.7500068419163685e-71), np.float64(-0.008867002273057624), np.float64(-0.18404441770827873), np.float64(0.6090112866040944), np.float64(4440.5225), np.float64(-0.7851551525974355), np.float64(0.08051353719354878), np.float64(9.5183)]


## Technical Approach

The objective is to maximize each of the eight black box functions.  Sounds easy but it isn't.

We are limited to one query a week and only about 13 total queries, so each query counts.

My technical approach is a combination of (a) heuristics (b) Bayesian optimization. (c) Software engineering.   I've recorded one Jupiter notebook per week to capture progress.  So far, it has gone as follows

Week 1: Heuristic - go to center of search space, find vector pointing to largest known value.  Continue on this vector to the edge of the search space.  This exploits the old engineering saying: "solutions are always at the constraints"

Week 2: Bayesian optimisation - I fit a surrogate function and used the UCB (Upper Confidence Bound) sampling function, then chose the max of this function.  I tuned the UCB to favour exploration over exploitation.

Week 3:  After being only partially satisfied with the surrogate functions, I became more aggressive about forcing greater exploration of the search space.  I gridded the search space and then chose grid points that were the furthest from the existing measurements--a sort of "furthest neighbor" approach.  This tended to create query points like  (0,1,0,0,1) out at the edges of the search space.  Most of these were fruitless, but at least I now know that we are unlikely to have a hidden global optimum out at the edges of the search space.  This is particularly a concern in the higher dimensional functions

Week 4:  Technical approach TBD

As the weeks have gone by, I've converged on the following technical approach:

(1) Mostly use standard **Bayesian Optimization** as described in https://bayesoptbook.com/book/bayesoptbook.pdf and many other sources.  Bayesian optimisation was designed to balance exploration and exploitation in settings where new queries to the underlying function are expensive and rare.

(2) Apply **input and output transformations** when the problem calls for it, for example due to wildly varying outputs, or input features that cause wild swings in output.  This is similar to the approach taken by Huawei in https://valohaichirpprod.blob.core.windows.net/papers/huawei.pdf.  These are implemented by implementing the abstract class Transformation for the functions that use transformations.

(3) In some cases, **segment the search space into a region of interest** or "Trust Region" so that the Bayesian optimization surrogate function fitting problem is not overly influenced by far-away low-lying points. This approach is described, for example, in this paper from JetBrains https://valohaichirpprod.blob.core.windows.net/papers/jetbrains.pdf.  Currently this is a work-in-progress in the repo.

(4) In early weeks, or whenever a region of interest appears exhausted I forced a highly exploratory search by selecting a point as far away as possible from existing points, a "Farthest Point Sampling" approach.


## Required libaries
See `requirements.txt`.  Pretty standard machine learning stuff including `numpy`, `matplotlib`, `pandas`, `scikit-learn`.  `plotly`is used for interactive visualization of lower dimensional functions.  This is an excellent library because it allows you to spin figures around in order to inspect how the surrogate and the original data points overlap.  `tensorflow`and `torch`were used for experimentation and likely to be removed.

## repo structure
Likely to change over the next few weeks because the repository is in real need of re-factoring:

- `setup_venv.py` - Running this first will set up a fresh, isolated virtual environment with the packages listed in `requirements.txt`.  Usage (on a mac): `> python3 setup_venv.py 3.13`, followed by `source .venv/bin/activate`.  This isolates "dependency hell" hiccups, especially with bleeding edge packages

- `ìnitial data/` - Initial data provided to us for each function - obsolete and likely to remove soon

- `measurements/`
    - `latest/` - the latest data, incorporating all known measurements
    - `wk01/`... `wk0n/` - documents incoming data as I received it in the text files `new_inputs.txt`and `new_outputs.txt``
        - `function_1/`...`function_8` contain inputs and outputs for that function, containing the latest updates, and stored as `numpy` arrays for inputs and ouputs

The contents of `measurements/latest` are identical to those of `measurements/wk0n` for the highest `n`.

File in repo root are as follows:

- `update_measurements.ipynb` - Ingests fresh data and writes the directories described above.  Usage instructions inside the notbook

- `wk01_xxx.ipynb`...`wk0n_xxx.ipynb`- Notebooks for first few weeks.  

Likely to be re-factored in the next few weeks according as follows

TODO
---
- remove `tensorflow` and `torch` dependencies
- Numerous utilities are used across notebooks.  Centralize them into a shared package to avoid duplication across notebooks
- Consider bringing function data and manipulation methods (write to storage, etc) in an object.  Current functional implementation is cumbersome.



