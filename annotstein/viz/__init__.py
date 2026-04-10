"""Interactive visualization helpers for annotstein analysis outputs.

All functions in this module are thin wrappers around `plotly` and are
available only when the ``viz`` optional dependency group is installed::

    pip install annotstein[viz]

Each function accepts the plain-dict output produced by the analysis CLI
commands (or their Python equivalents) and returns a
:class:`plotly.graph_objects.Figure`.  Call ``.write_html(path)`` on the
returned figure to save an interactive HTML file.
"""
