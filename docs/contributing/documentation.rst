Documentation
=============

You are welcome to extend sampik's documentation! All help is appreciated.

Using reStructedText
--------------------

* `Basics of rST <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
* `Documenting C++ with rST <https://www.sphinx-doc.org/en/master/usage/domains/cpp.html>`_

Building a local copy of the docs
---------------------------------

First, create a virtual environment at ``./.venv``:

.. code-block:: console

	$ python3 -m venv .venv

Activate the virtual environment:

.. code-block:: console

	$ source .venv/bin/activate

Install the required packages for building the docs:

.. code-block:: console

	$ pip install -r docs/requirements.txt

Build the documentation:

.. code-block:: console

	$ make -C docs html

You can then open the documentation in your favorite browser:

.. code-block:: console

	$ $BROWSER docs/_build/html/index.html
