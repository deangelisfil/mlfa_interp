import matlab.engine
from pathlib import Path


def initialize_matlab_engine(parent_dir: int):
    eng = matlab.engine.start_matlab()
    path_spinterp = Path.cwd().parents[parent_dir] / "spinterp"
    eng.cd(str(path_spinterp))
    eng.addpath('spinterpv5/', nargout=0)
    eng.eval("path = py.pathlib.Path.cwd().parent.absolute();"
             "py.sys.path().append(py.str(path));", nargout=0) # add path to the Python path
    return eng

