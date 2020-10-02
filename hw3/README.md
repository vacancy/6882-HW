# Coding problem

1. Do I agree with the bullet points?

Mostly agree, except the third bullet point for the second installment.

> Conversely, the actual time that it took to run (let alone develop) the training pipeline most likely exceeded the time that it took you to write your own custom heuristics.

For level 1 to level 3, our heuristic learning algorithms took less than 20 seconds to train and performs as well as A star uniform. Given twenty seconds, we might design a heuristic but not necessarily work well.  

## 2. How I would write the SAR domain in PDDL differently


```
### LEVEL 1 ###

# Means #
Approach         Train Time    Duration    Num Steps    Num Nodes    Successes
-------------  ------------  ----------  -----------  -----------  -----------
astar_hadd      7.15256e-07   0.0331247         11.6         13.8            1
astar_hff       1.19209e-06   0.0329892         11.6         14.2            1
astar_hmax      9.53674e-07   0.0451015         11.6         21.6            1
astar_uniform   2.14577e-06   0.212209          11.6        148              1

### LEVEL 2 ###

# Means #
Approach         Train Time    Duration    Num Steps    Num Nodes    Successes
-------------  ------------  ----------  -----------  -----------  -----------
astar_hadd      1.90735e-06   0.0500237         13.1         16.9            1
astar_hff       1.19209e-06   0.0490153         13.1         15.9            1
astar_hmax      7.15256e-07   0.0687376         12.9         25.9            1
astar_uniform   2.14577e-06   0.323025          12.9        227.3            1

### LEVEL 3 ###

# Means #
Approach         Train Time    Duration    Num Steps    Num Nodes    Successes
-------------  ------------  ----------  -----------  -----------  -----------
astar_hadd      2.14577e-06   0.0872813         11.1         13.7            1
astar_hff       1.19209e-06   0.0854214         10.9         13.4            1
astar_hmax      9.53674e-07   0.10667           10.9         20.2            1
astar_uniform   9.53674e-07   0.200027          10.9        142.2            1

### LEVEL 4 ###

# Means #
Approach         Train Time    Duration    Num Steps    Num Nodes    Successes
-------------  ------------  ----------  -----------  -----------  -----------
astar_hadd      9.53674e-07    0.136458         11.5         18.1            1
astar_hff       9.53674e-07    0.161621         11.3         19.9            1
astar_hmax      2.14577e-06    0.305137         11.3         50.9            1
astar_uniform   1.90735e-06    2.15318          11.3       1152.7            1

### LEVEL 5 ###

# Means #
Approach         Train Time    Duration    Num Steps    Num Nodes    Successes
-------------  ------------  ----------  -----------  -----------  -----------
astar_hadd      9.53674e-07    0.775016         23.5        161.4          1
astar_hff       9.53674e-07    2.66935          14.6        552.5          0.8
astar_hmax      1.19209e-06    6.11781           7.4       1284.8          0.5
astar_uniform   1.66893e-06    8.99479           2.5       4619            0.2

### LEVEL 6 ###

# Means #
Approach         Train Time    Duration    Num Steps    Num Nodes    Successes
-------------  ------------  ----------  -----------  -----------  -----------
astar_hadd      1.19209e-06     2.43737         28.4        516.4          1
astar_hff       1.66893e-06     3.6769          14.3        749.8          0.7
astar_hmax      9.53674e-07     7.93027          7.4       1757.3          0.4
astar_uniform   9.53674e-07     9.44697          1.3       4799.7          0.1
```