
S.D. Peckham, August 2022

# Converting the LSTM Model to a Python Package

While a Python model does not need to be distributed as a Python package to be
run in the NextGen framework, there are several advantages to doing so.
Detailed instructions on how to create a Python package are given in these
online articles:
[Packaging Python](https://packaging.python.org/en/latest/tutorials/packaging-projects/),
[A Practical Guide to Using Setup](https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/).  The method described in the first article is now preferred, but the one in the second article still works and is widely used.
Also see:
* ngen/extern/test_bmi.py.
* https://setuptools.pypa.io/en/latest/references/keywords.html
* https://towardsdatascience.com/setuptools-python-571e7d5500f2

## What is a Python Package?

A Python package is a directory with Python files and a file with the name `__init__.py`.
The file `__init__.py` must exist, but can be empty, and typically is.  Every directory
inside of the Python path which contains a file named `__init__.py`, is treated as a
package by Python.  Each Python file in the package directory is viewed as a Python
**module**.  The name of the package directory may be referred to as the "package name"
or "distribution name" and the package will be imported into Python using this name.
It should be short and somewhat descriptive, often a shortened version of the project
or GitHub repo name. The package directory should be contained within a parent
"project directory" (often the project repo) that contains other files and folders
related to the project.  There are no rules about the name of the project directory
vs. the name of the corresponding Python package it contains, and they may be the
same.  The general layout of the project folder typically looks like:
```
project_name/
    - docs/
    - LICENSE
    - notebooks/
    - package_name/
        - __init__.py  b (often empty)
        - __main__.py
        - module_name1.py
        - module_name2.py
    - pyproject.toml
    - README.md
    - setup.cfg   (optional) 
    - setup.py
    - tests/
```
However, sometimes the `project_name` folder will contain a folder called "src"
which contains the package folder.


## LSTM as a Python Package

* Copied lstm project/repo folder into `ngen/extern/lstm`.
  - Could change project folder name to: `lstm_py`, or `lstm_model`.
* Created a package folder called "lstm" in the project folder (also lstm).
  - this will allow "import lstm"
* Moved Python source code files into package folder, "lstm".
  - `bmi_lstm.py`
  - `nextgen_cuda_lstm.py`
  - `run_lstm_with_bmi_v2.py`
* Created empty file `__init__.py` in the package folder.
* Created a file called `__main__.py` in the package folder.
  - allows model to be run with:  `python -m lstm`
* Created a file `run_lstm_with_bmi_v2.py` in the package folder.
  - Started with original file:  `run-lstm-with-bmi.py`
  - Renamed to avoid hyphens in filename (problematic)
  - Moved commands into an "execute" function.
  - New execute function is called in `__main__.py`.
  - Changed "`import bmi_lstm.py`" to: "`import lstm.bmi_lstm.py`"
  - Modified some of the print statements a bit.
* Created a basic setup.py file in project folder.  (But no longer preferred method.)
* LSTM uses files from the following 3 folders: `bmi_config_files`, `data`, and
`trained_neuralhydrology_models`.  Note that I did not put these in the lstm
package folder, but they are needed and could possibly be considered as and
treated as "package data".
 

## Installing the "lstm" Package into venv

Assume that a Python virtual environment named "venv" has already been installed
in the ngen folder.  We must activate this environment before installing lstm
and its dependencies, which are listed in the `setup.py` file.
```
% cd ngen
% source venv/bin/activate
% cd lstm
% pip install -e .
```

## Check the Installation

Once you have successfully installed the lstm package in venv (and
activated that environment) you should be able to launch a Python
session from any directory and still import lstm.  You should not need
to edit your Python path.
```
% python
>>> import lstm

% cd ..    (should still work from here)
% python
>>> import lstm
```

## Run LSTM at OS Command Prompt
```
% source venv/bin/activate
% python -m lstm    (uses `__main__.py`)
```

## Important Notes

* LSTM reads training data from a pickle file with extension ".p" that includes
an object created by xarray v.0.16.0 (or earlier).  So even though the xarray
package is not imported by any lstm module, that version of xarray must be
installed to avoid the error message:
*AttributeError: 'Dataset' object has no attribute '_file obj'*
See: [Error using rioxarray](https://stackoverflow.com/questions/66432884/error-using-rioxarray-in-jupyter-notebook)
See: [Loading a pickled xarray object](https://github.com/pydata/xarray/discussions/5642).

* The LSTM Python model uses Path from the pathlib package is several places.
This isn't necessary when installed as a package, and I've added a boolean
flag to the code called `USE_PATH` that shows how the code can work with or
without using "Path".

* The `bmi_cfg_file` argument to the BMI initialize function defined in
`bmi_lstm.py` should be of type "str", but as originally written the
initialize function is passed an object of type Path (from pathlib).
Modified to apply Path() within initialize instead.

* Some of the lstm modules contain the line:  "from torch import nn",
but "nn" is not used anywhere.




