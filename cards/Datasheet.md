# Datasheet for black box functions

**Motivation**: This is a dataset of toy functions used in the Imperial College, London "Black Box optimization challenge."  The goal is to use a small nubmer of samples to maximize eight different functions.  This is a synthetic dataset consisting of datapoints given by Imperial, augmented with 13 exploratory points that I generated in one of three ways
1.  Furthest point exploration - in the initial weeks 
2. Bayesian Optimization using a globally fitted Gaussian process - until about week 7
3. Bayesian Optimization using a Gaussian Process fitted in a neighborhood of high points from week 8 onwards

**Contents**: Data is organized in the ```measurements``` folder as follows

```text
project-root/
├── measurements/
│   ├── latest/                         # most up-to-date dataset
│   │    ├── function_1/                # inputs and outputs of first function as npy arrays - function_2... similar
│   │    │      ├── inputs.npy
│   │    │      ├── outputs.npy
│   │    ├── function_2/
│   │    ├── function_3/
│   │    ├── function_4/
│   │    ├── function_5/
│   │    ├── function_6/
│   │    ├── function_7/
│   │    ├── function_8/
│   └── wk00                            # historic - original clean dataset provided by Imperial.  Same contents as latest/
│   └── wk01                            # historic - original dataset + week 1 exploratory datapoints
│   └── wk02                            # historic - week 1 + week 2 exploratory datapoints
│   └── wk03
│   └── wk04
│   └── ...
```


Functions are as follows:

f_1():  R2 --> scalar output
f_2():  R2 --> scalar output
f_3():  R3 --> scalar
f_4():  R4 --> scalar
f_5():  R4 --> scalar
f_6():  R5 --> scalar
f_7():  R6 --> scalar
f_8():  R8 --> scalar


**Collection process**: As the directory structure above implies, the data was accumulated over a period of about 13 weeks, one additional datapoint for each of the eight functions each week.

**Preprocessing and uses**: All data is unitless.  Inputs are on the internal [0,1] and outputs are unlimited.  The stored data is raw, ie no transformations are applied to the files here.

**Distribution and maintenance**: This dataset may be freely used.  ```mucdat``` maintains it. In the unlikely case where you would like to write to it, please submit a pull request.  

