## Automatic AIfES for Gossiping Learning - An Intrusion Detection System (IDS) Use-Case
The project consists of an implementation of gossip learning using Node.js, PyTorch, and Docker.

To start the project, Docker must be installed on the system. The neural network consists of 8 nodes, instantiated as Docker containers and defined in the docker-compose.yml file. The ```python-bridge``` library was used to execute Python code within Node.js. For communication between nodes, the ```libp2p``` library was used.

To start the learning process, run the following command from the terminal:
```bash
docker build --tag gossip-aifes
docker run gossip-aifes
```

To stop all the containers
```bash
docker down
```

To Do:
- [ ] Define the model merging process, if this should be done by Node.js or by the C-Code. To change on line 108 of ```gossip_learning.js```.
- [ ] Define the protocol for weights exchange, currently it sends the weights to a random node as written on line 225 of ```gossip_learning.js```.