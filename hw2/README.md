# Answers for Coding Question's written questions

Jiayuan Mao and Zhutian Yang collaborated to finish this Coding Question.


## 1. We implemented the three approaches described in the question

*supervised_policy*: We trained a decision tree classifier for predicting the next action given state. The predicted actions are taken during testing. The training data come from the pairs of state and next action planned by the Uniform Cost Search algorithm for every problem in the training environment. We applied PropositionalFeaturizer to the states and TabularFeaturizer to the actions. For states that didn't appear in the training data but appear during testing, we randomly select an action from the available actions.

*supervised_heuristic*: We trained a multi-layer perceptron regressor for predicting the heuristic value of a state, i.e., the number of remaining steps to the goal state, given the state. The predicted heuristic value is used by A* Star algorithm for planning during testing. The data were collected in the same way as the previous approach. We applied SARMinimalStateFeaturizer to the states.

*qlearning_heuristic*: We trained a multi-layer perceptron regressor for predicting the heuristic Q(s,a) value given a state-action pair. The predicted Q values is used as a heuristic by A* Star algorithm for planning during testing. During training, we first do tabular Q learning in the training environments; at each step, we extract an epsilon greedy policy (epsilon = 0.1) from the current Q values, collect rewards, and update Q value using temporal learning formula with learning rate 0.5 and discount factor 0.9. Using the learned Q table and corresponding (s, a) pairs, we fit a RLE regressor to predict the Q value of unseen state-action pairs. We applied SARMinimalStateFeaturizer to the states and TabularFeaturizer to the actions.

## 2. How do the three approaches compare to each other?

Test environment accuracy over all levels: supervised_heuristic > qlearning_heuristic > supervised_policy

Training time: constant for qlearning_heuristic because of the fixed number of simulated steps for Q-learning; the training time for supervised_policy and supervised_heuristic increases with the size of the state space (which increases with level).

## 3. Interesting insights?

Looking at the statistics for level four, we found that the while astar_uniform has 0.9 success rate, supervised_heuristic achieved full success rate. This is surprising because the data used for training the regressor is collected using the astar_uniform planning on the training problems. This implies that learning a heuristic function is more generalizable than learning the policy directly.

## 4. Experimental details

Here we show the the means of statistics.

```
### LEVEL 1 ###

# Means #
Approach                Train Time    Duration    Num Steps    Num Nodes    Successes
--------------------  ------------  ----------  -----------  -----------  -----------
random                 9.53674e-07   0.0670754        250            0            0
astar_uniform          2.14577e-06   0.206022          11.6        148            1
supervised_policy      4.10566       0.017256          35.5          0            0.9
supervised_heuristic   5.43012       0.0766504         11.6         31.4          1
qlearning_heuristic   34.0647        0.745613          11.6        154            1


### LEVEL 2 ###

# Means #
Approach                Train Time    Duration    Num Steps    Num Nodes    Successes
--------------------  ------------  ----------  -----------  -----------  -----------
random                 1.90735e-06   0.0811692        250            0            0
astar_uniform          9.53674e-07   0.35642           12.9        227.3          1
supervised_policy     16.4706        0.0786166        131.5          0            0.5
supervised_heuristic  18.4331        0.0820764         12.9         40.8          1
qlearning_heuristic   35.9111        0.669516          12.9        213.7          1


### LEVEL 3 ###

# Means #
Approach                Train Time    Duration    Num Steps    Num Nodes    Successes
--------------------  ------------  ----------  -----------  -----------  -----------
random                 9.53674e-07   0.0724581        250            0            0
astar_uniform          9.53674e-07   0.196898          10.9        142.2          1
supervised_policy     12.9152        0.107918         202.2          0            0.2
supervised_heuristic  18.6148        0.0868547         10.9         43.4          1
qlearning_heuristic   30.555         0.408206          10.9        144.1          1


### LEVEL 4 ###

# Means #
Approach                Train Time    Duration    Num Steps    Num Nodes    Successes
--------------------  ------------  ----------  -----------  -----------  -----------
random                 9.53674e-07   0.0615966        234.2          0            0.1
astar_uniform          9.53674e-07   2.33962            9.8       1144.3          0.9
supervised_policy     65.9997        0.140982         250            0            0
supervised_heuristic  68.0727        1.52979           11.3        651.1          1
qlearning_heuristic   27.3887        3.41399            8.5        842.5          0.8


### LEVEL 5 ###

# Means #
Approach                Train Time    Duration    Num Steps    Num Nodes    Successes
--------------------  ------------  ----------  -----------  -----------  -----------
random                      9.537e-07  0.0670822      250            0            0
astar_uniform               1.907e-06  9.25208          1.2       4363.9          0.1
supervised_policy         472.856      0.1643         250            0            0
supervised_heuristic      472.953     10.0271           0         3732.3          0
qlearning_heuristic        32.7663     9.43929          1.2       2009.4          0.1


### LEVEL 6 ###

# Means #
Approach                Train Time    Duration    Num Steps    Num Nodes    Successes
--------------------  ------------  ----------  -----------  -----------  -----------
random                       2.146e-06 0.0816676        250          0              0
astar_uniform                9.537e-07 9.53772            1.3     3767              0.1
supervised_policy          443.557     0.13611          250          0              0
supervised_heuristic       447.512    10.0243             0       3676.7            0
qlearning_heuristic         35.707    10.0203             0       1889.5            0
```


# Course Dev Questions

Jiayuan Mao and Zhutian Yang collaborated on the course development question.

In this section, you are going to play with and compare two other planning algorithms:
the Upper Confidence Tree (UCT) algorithm and the Value Iteration (VI) algorithm.

1. Implement a basic version of UCT and VI, describe the hyperparameters for both algorithms.

*Answer*: Please refer to the following repo as the reference implementation.
In our implementation, the UCT algorithm has the following hyperparameters:

- `max_steps`: the max depth of the search tree.
- `num_search_iters`: number of UCT iterations.
- `replanning_interval`: after a certain number of environment steps, we are going to replan.
- `gamma`: discount factor.
- `timeout`.

The Value Iteration algorithm has the following hyperparameters.
- `gamma`: discount factor.
- `epsilon`: epsilon for early stopping the value iteration.

2. Test both algorithms on the level 1 of the save-and-rescue environment, report the time spent on the search and the success rate for both algorithms.
_Hint: you may want to use a longer timeout for both algorithms as they run slow._

*Answer*:

```
# Means #
Approach         Train Time    Duration    Num Steps
-------------  ------------  ----------  -----------
astar_uniform   1.19209e-06    0.129424            9
uct             1.43051e-06   32.9664             25
value_iteration 2.14577e-06   50.5765              9
```
Actually it took 10.88 seconds for Value iteration to converge, the extra 39.362 seconds was spent on translating the state space.

3. Both UCT and A-Star with uniform heuristic do not use any informative heuristic functions. Compare their performances and describe your findings.

*Answer*:
A-Star runs a magnitude faster than UCT. This is possibly because:

- A-Star assumes a single goal state, but UCT was designed for achieving max accumulated rewards. Thus, UCT in general solves a more difficult problem.
In the fixed horizon case, the UCT algorithm keeps track of the number of remaining steps, making the search graph H times bigger, where H is the planning horizon.
- In our implementation, at each iteration, UCT starts from the starting state.

4. Show the complexity of the Value Iteration algorithm. Why is it slow? What's the assumption difference between Value Iteration and A* (or other search algorithms?)

*Answer*: The Value Iteration takes O(kNA) time. k is the number of iterations. N is the number of states. A is the number of actions.
The key difference between VI and A* is that VI is capable of handling non-deterministic environments. That is, the transition function can be non-deterministic.
In such case, the A* algorithm can not produce the expected value at each state.
Moreover, value iteration is designed for the general reward-maximization problem, whereas A* terminates after finding a path to a single goal state.

