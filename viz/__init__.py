"""Presentation layer for the Streamlit app.

Nothing in here is imported by ``hfmarl``. The dependency runs one way only:
the visualiser reads the research package, never the reverse, so a figure can
never quietly change a number that a gate is measured against.
"""
