# Model Card

**Overview:** This model implements Bayesian Optimization with Gaussian Process model and UCB (Upper Confidence Bound) acquisition function.  It has options for
- Local trust regions with local Gaussian Process fits
- "Furthest point" sampling for maximum exploration, ie find the point that is furthest from any existing point

**Intended use**: The model was originally developed for the Imperial Colleage Machine Learning Professional Certificate "Black Box Challenge."  

**Details**: During the competition, the approach changed as follows
1. Furthest point exploration - in the initial weeks for maximum exploration
2. Bayesian Optimization using a globally fitted Gaussian process - until about week 7
3. Bayesian Optimization using a Gaussian Process fitted in a neighborhood of high points - from week 8 onwards

**Performance**: The goal of the model is to maximize each of eight functions (see their details in the [Datasheet](./Datasheet.md) and [README](../README.md)).  As such, the performance metrix is the maximum value achieved on each of the eight functions.

**Assumptions and limitations**: The model was originally developed for a competition in which the search space is on the interval [0,1] and for two to eight dimensions.  The [0,1] interval is used in several places in the code.  Using on search spaces that are not in the interval [0,1] requires either scaling your inputs to the [0,1] interval, or it requires updatin the code to use arbitrary [x_min,x_max] intervals.

**Ethical considerations**: No major ethical considerations for this model are known at this time.  By making the approach transparen, I hope/expect to collaborate with colleagues for the last few weeks fo the competition.
