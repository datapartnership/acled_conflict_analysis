from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mobilyze")
except PackageNotFoundError:
    # package is not installed
    pass
