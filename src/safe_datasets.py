import sys
import types

module_name = "sklearn.datasets._svmlight_format_io"

if module_name not in sys.modules:
    module = types.ModuleType(module_name)
    module.dump_svmlight_file = None
    module.load_svmlight_file = None
    module.load_svmlight_files = None
    sys.modules[module_name] = module

from sklearn.datasets import fetch_openml
