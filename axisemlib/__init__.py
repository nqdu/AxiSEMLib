"""AxiSEM wavefield processing and SPECFEM coupling tools."""

__version__ = "3.0.0"
__all__ = ["AxiBasicDB"]


def __getattr__(name: str):
    if name == "AxiBasicDB":
        from .database import AxiBasicDB

        return AxiBasicDB
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
