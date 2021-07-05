
Installation
===================

NeuralXC can be obtained free-of-charge from our Github repository `here <https://github.com/semodi/neuralxc>`_.

To install NeuralXC using pip, after cloning the repository to your local machine, navigate into the root directory of the repository and run::

  sh install.sh

So far, NeuralXC has only been tested on Linux and Mac OS X.

To check the integrity of the installation, unit tests can be run with::

  pytest -v

in the root directory. The installation succeeded if the pytest summary shows no failed test::


  =========  25 passed, 19 warnings in 47.56s =========


In case of reported failures, try re-downloading and re-installing the library. If problem persists,
please raise an issue on the github repository.
