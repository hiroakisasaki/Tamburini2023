# CSA_OptMatcher

This code is devoted to help paleographers and linguists in the decipherment of lost script/languages. Please, refer to the cited paper for a complete explanation of the general goals, configurations, parameters and datasets.

This package requires the installation of another package I developed, namely [EditDistanceWild](https://github.com/ftamburin/EditDistanceWild).

## For repeating the experiments on the paper:
- First you can train the system with, for example
```
    python3 CSA_OptMatcher.py -c data/linear_b-greek.cog -f data/FixNULL -n 2 -m 1 -d cuda 1> LOG 2> LOG_E
```
monitoring the training by looking at the *stderr* log 'LOG_E'. 

- The program prints on *stdout* all the tested solutions by the CSA process, thus, sorting them using the file 'LOG' we can get the lowest-energy solution and test it by the command
```
    python3 CSA_OptMatcher.py -c data/linear_b-greek.cog -f data/FixNULL -n 2 -m 1 -d cuda -o "[34, 44, ..., 20, 72]"
```

Tested on:
- python 3.7
- pytorch 1.10.2

If you find this code useful in your research, please cite:
```
@InProceedings{Tamburini:2023:ACL-CAWL,
  author    = {Tamburini, Fabio},
  title     = {Decipherment of Lost Ancient Scripts as Combinatorial Optimisation using Coupled Simulated Annealing},
  booktitle = {Proceedings of the 1st ACL Workshop on Computation and Written Language (CAWL)},
  month     = {July},
  year      = {2023},
  address   = {Toronto, Canada},
  publisher = {Association for Computational Linguistics}
}
```

In case of problems contact me at <fabio.tamburini@unibo.it>.
