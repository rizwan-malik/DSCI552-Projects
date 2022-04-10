## Homework 7 - Hidden Markon Models
Use a Hidden Markov Model to figure out the most likely sequence of values given an observation sequence. At timestep *t* value *v* can be any value from a given domain space.

Consider a variable ***x*** with domain ***{1, 2, 3 ... 10}***. Let ***vt*** be the value of ***x*** at timestep ***t***. ***vt+1*** is equal to ***vt- 1*** or ***vt + 1*** with probability ***0.5*** each, except when ***vt = 1*** or ***vt = 10***, in which case ***vt+1 = 2*** or ***vt+1 = 9***, respectively. At each timestep ***t***, we also get noisy measurements of ***vt***. That is, ***vt -1***, ***vt*** or ***vt + 1*** can be returned with equal probabilities.

#### Deliverable:
Use a Hidden Markov Model to figure out the most likely sequence of values ***v1 v2 ... v10*** when the observation sequence is ***8, 6, 4, 6, 5, 4, 5, 5, 7, 9***. At timestep ***t = 1***, ***v1*** can be any value in ***{1, 2, 3 ... 10}*** with equal prior probabilities.
