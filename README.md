# PyTorchUtils
Some utilities for working with PyTorch and torms3's DataProvider for connectomics data

This package is meant to be EXTREMELY lightweight. It represents a notetaking system rather than a stable updated codebase. This decision was motivated by observing that different experiments often require tweaks that make clean code abstraction difficult. Instead, this tries to abstract a core set of functions (training, inference), and record everything else as separate files. These files are often extremely redundant, but they also allow us to keep precise records and replicate experiments without difficulty.

|Required Packages|
|:-----:|
|[PyTorch](http://pytorch.org/)|
|[DataProvider](https://github.com/torms3/DataProvider/tree/refactoring)|
|[numpy](http://www.numpy.org/)|
|[h5py](http://www.h5py.org/)|
|[tensorboardX](https://github.com/lanpa/tensorboard-pytorch)|

