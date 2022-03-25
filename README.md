# Masters Project: Distributed Optimization in Multi-agent Systems with Coupling Constraints
An effective strategy to accelerate the numerical solving of constrained optimization problems is to
split the global problem into a set of smaller problems, each of which can be solved in a different parallel
process. Since the resulting local problems may be coupled to each other through global constraints,
a number of distributed schemes exist to enforce coordination between these processes through an exchange of information. However, these methods often require many iterations to converge, demanding
extensive communication between agents, which may be impractical in a real physical network. To
address this limitation, we propose connected dominating set sweeping methods, which exploit the
structure of a network of agents to solve optimization problems in a fully decentralized manner. As
well as constructing a series of local problems that are significantly simpler than those of competing
schemes, these algorithms can also be implemented with an efficient communication protocol, making
them ideal for real cyber-physical systems. We test the methods on a set of dynamic optimization
problems, demonstrating impressive performance in comparison to alternative strategies.

This repository contains the code developed throughout the project [external active set method](https://github.com/alexpopov1/masters-project/tree/main/ExternalActiveSet) (see folders 'ExternalActiveSet', 'Formation' and 'Ring') as well as a PDF versions of the final thesis and the accompanying presentation slides (see 'Thesis').
