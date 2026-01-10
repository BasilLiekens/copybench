# Copybench

This repository contains some simple benchmark code that is used for [Scipy issue #24340](https://github.com/scipy/scipy/issues/24340). It consists of a simple scripts that runs various "copy-and-transpose" techniques to see the impact of these different memory copying strategies. Since it is code for a Scipy issue, a lot of the code here is copied straight from the [Scipy repository](https://github.com/scipy/scipy).

## Folder structure
The source code for the package can be found under `src`. This can be installed through `pip install .` which will automatically install all required dependencies. The main script `main.py` is located and can be found under `scripts`.

Running the main script then is simply calling `python scripts/main.py`.
