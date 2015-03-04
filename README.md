# viztricks
Recipes and helper functions for plotting with Python + matplotlib

To install, clone this repository and run:

    python setup.py install

This package depends on numpy, scipy, and matplotlib.
The `pca_ellipse` function relies on scikit-learn as well, but this is an
optional dependency.
To create animated GIFs with `FigureSaver`, imagemagick must be installed
with the `convert` program on the path.

To run the test suite:

    ./run_tests.sh

Testing requires python packages nose and nose-cov.
