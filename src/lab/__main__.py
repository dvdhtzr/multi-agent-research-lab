"""
Allows `python -m lab` to invoke the CLI entry point.

C# parallel: the equivalent of a console application's Program.cs Main() method
being the single entry point, callable from the command line.
"""
from lab.cli import main

if __name__ == "__main__":
    main()
