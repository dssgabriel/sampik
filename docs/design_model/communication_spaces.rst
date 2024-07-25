Communication Spaces
====================

A Communication Space is the underlying message passing mechanism that sampik uses to perform communications, e.g. MPI or NCCL.

Communication Spaces are used to specialize a ``sampik::Handle`` for a given backend. Handles wrap a backend's "communicator" objects and provide generic interfaces for retrieving information such as the size or rank of the underlying communicator.

``sampik::MPI``
---------------

``sampik::MPI`` is a type of that "implements" the ``CommunicationSpace`` concept, representing communications that use MPI. 

``sampik::NCCL``
----------------

``sampik::NCCL`` is a type of that "implements" the ``CommunicationSpace`` concept, representing communications that use NCCL. 

