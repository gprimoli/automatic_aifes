## Automatic AIfES for Gossiping Learning - An Intrusion Detection System (IDS) Use-Case

The project consists of an implementation of gossip learning using Node.js, PyTorch, and Docker.

To start the project, Docker must be installed on the system. The neural network consists of 8 nodes, instantiated as Docker containers and defined in the docker-compose.yml file. The `python-bridge` library was used to execute Python code within Node.js. For communication between nodes, the `libp2p` library was used.

To start the learning process, run the following command from the terminal:

```bash
docker compose build
docker compose up
```

To stop all the containers

```bash
docker compose down
```

To Do:

- [ ] Define the model merging process, if this should be done by Node.js or by the C-Code. Currently it is demened to a Python script. To change on line 59 of `gossip_learning.js` in case this behaviour is not okay.
- [ ] Define the protocol for weights exchange, currently it sends the weights to a random node as written on line 163 of `gossip_learning.js`. Notice that this apporach also implies that a custom node is selected for the running of the first gossip of the model.
   =======
