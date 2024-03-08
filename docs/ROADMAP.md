# Roadmap


## 1. Initial work

- [ ] Rank-1 contiguous views
- [ ] GPU-aware MPI implementations
- [ ] Basic P2P comms: `send`, `recv`
- [ ] Necessary routines: `initialize`, `finalize`, `wait`, etc.


## 2. Expand with non-contiguous views

- [ ] Rank-1 non-contiguous views via either of:
  - [ ] `deep_copy` into a "packed" view
  - [ ] constructing custom `MPI_Datatype`s
- [ ] Richer P2P comms using overloads


## 3. Handle multi-dimensional views and begin work on collectives

- [ ] Non GPU-aware MPI implementations: `deep_copy` on host and perform a "traditional" communication
- [ ] Multi-dimensional contiguous views, at least for P2P communications
- [ ] Collectives, at least `reduce`


## 4. Enrich the API with less common communication needs

- [ ] Enumeration for specific communications semantics: bufferized, synchronous, etc.
- [ ] Richer `wait` and `test` APIs


## 5. Utility functions for a more idiomatic use

- [ ] Create views from a received message
- [ ] Calls from parallel regions


## 6. Expand collectives and prospect for new features

- [ ] Richer collectives: `broadcast`, `gather`, `scatter`, `all_to_all`, etc.
- [ ] Evaluate what else is needed and not yet supported at this point
