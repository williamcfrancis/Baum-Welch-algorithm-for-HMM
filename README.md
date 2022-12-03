# Baum Welch algorithm for HMM
The Baum–Welch algorithm is a special case of the expectation–maximization algorithm used to find the unknown parameters of a hidden Markov model (HMM). It makes use of the forward-backward algorithm to compute the statistics for the expectation step.

The following scenario is considered and solved using the algorithm:

Assume that we are tracking a criminal who shuttles merchandise between Los Angeles (x1) and New York (x2). The state transition matrix between these states is
![image](https://user-images.githubusercontent.com/38180831/205466260-7a5fb96f-247c-4e34-abc2-58745fbf8d4e.png)
e.g., given the person is in LA, he is likely to stay in LA or go to NY with equal probability. We can make observations about this person, we either observe him to be in LA (y1), NY (y2) or do not observe anything at all (null, y3).
