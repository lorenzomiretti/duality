# UL-DL Duality for Cell-free Massive MIMO with Per-AP Power and Information Constraints

This is a code package related to the following scientific article:

Lorenzo Miretti, Renato L. G. Cavalcante, Emil Björnson, Slawomir Stan\'nczak, “UL-DL Duality for Cell-free Massive MIMO with Per-AP Power and Information Constraints,” in *IEEE Transactions on Signal Processing*, vol. 72, pp. 1750-1765, 2024, doi: 10.1109/TSP.2024.3376809.

The package contains a simulation environment that reproduces the numerical results in the article. We encourage you to also perform reproducible research!

## Abstract of the article

We derive a novel uplink-downlink duality principle for optimal joint precoding design under per-transmitter power and information constraints in fading channels. The information constraints model limited sharing of channel state information and data bearing signals across the transmitters. The main application is to cell-free networks, where each access point (AP) must typically satisfy an individual power constraint and form its transmit signal using limited cooperation capabilities. Our duality principle applies to ergodic achievable rates given by the popular hardening bound, and it can be interpreted as a nontrivial generalization of a previous result by Yu and Lan for deterministic channels. This generalization allows us to study involved information constraints going beyond the simple case of cluster-wise centralized precoding covered by previous techniques. Specifically, we show that the optimal joint precoders are, in general, given by an extension of the recently developed team minimum mean-square error method. As a particular yet practical example, we then solve the problem of optimal local precoding design in user-centric cell-free massive MIMO networks subject to per-AP power constraints.

## Content of Code Package

The article contains 2 simulation figures, numbered 2-3. Figure 3 is composed by 2 subfigures, labelled a-b.

Figure 2 is generated by the Python script 
> local_vs_centralized_rate.py

Figure 3 is generated by the Python script 
> algorithm_convergence.py

The main routines used by the above scripts are collected in the Python script
> main_problem.py

Warning: The name of some variables may differ from the notation adopted in the paper. 

## Acknowledgements

This work recieved financial support by the Federal Ministry of Education and Research of Germany in the programme of “Souverän. Digital. Vernetzt.” Joint project 6G-RIC, project identification numbers: 16KISK020K, 16KISK030, and by the FFL18-0277 grant from the Swedish Foundation for Strategic Research.

## License and Referencing

This code package is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our original article listed above.
