from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("acled_conflict_analysis")
except PackageNotFoundError:
    # package is not installed
    pass

# Import the processing module so it's available when importing the package
from . import processing
from . import visuals
from . import extraction