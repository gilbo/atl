
# A Tensor Language

_(What should this be called?  Iver after Iverson?  ATL after APL?  Ivar for index-variable?  Something else entirely?)_

This is a working space for sketching a tensor language.  I am not placing a high degree of stress on portability right now.  Please schedule time with me to iron out various portability issues if you want to get this working on Linux or Windows, your machine etc.


## Early Notebooks

Most of the work has been sketched in notebooks and then copied into the `src/` files with some degree of editing.  Perhaps a final version should edit the notebooks up to match the source exactly for posterity's sake.

The notebooks *ought* to be read chronologically.  That order is recorded here:

* How-to Represent IRs
* Memoization of IRs
* A Tensor Language v0
* Wrapping Halide Part 1
* Wrapping Halide Part 2
* Compiling A Tensor Language v0

I will now briefly gloss each of these.


### How-to Represent IRs and Memoization of IRs

These notebooks generated `src/adt.py`, which is a tiny DSL designed to help compiler writers create type-checked Python class hierarchies out of descriptions that look like BNF grammars.  The grammar language syntax is called ASDL, from the Princeton Zephyr project.  It is used by the standard Python compiler to document Python's own IR.

There is probably a `pip` dependency here that isn't well documented.  Can you help teach me how to create an executable file of Python package dependencies?


### A Tensor Language v0

This is the main course of the notebooks so far.  It shows how to bootstrap up a language sketch using the ADT tools and Python hackery very quickly.  This is my overriding modus operandi with this project: figure out how to get the big ideas across without writing much code.  (see `src/atlv0.py` for the final result)


### Wrapping Halide, Part 1 & 2

This is an exercise in how to pare down a whole build-system and library wrapper into less than 500 lines of code using clever meta-programming and filesystem inspection/introspection tricks.  It exists to support a careful attempt to compile the tensor language efficiently.  (see `src/halide.py` for the result of this notebook, and `test_halide.py` for a very few test cases present)


### Compiling a Tensor Language v0

An experiment (quite long) in forcing the tensor language to compile down to Halide.  The results are collected in `src/atlv0_compile.py`









