"""Query workload generator (step 1 of the QuickSel pipeline).

Modules:
    java_random  - exact copy of java.util.Random, so query rectangles match the Java code
    data         - dataset generators (Gaussian) and the real-CSV loader
    queries      - range-query (workload) generators
    selectivity  - fast, vectorized true selectivity
    io           - read/write query files in the repo format
    generate     - command-line tool that ties everything together
    plot_queries - draw 2D queries as rectangles (like paper Figure 2)

Always run modules with ``python -m python_impl.workload.<module>`` from the
project root (never ``python io.py``), otherwise our ``io.py`` would hide
Python's built-in ``io`` module.
"""
