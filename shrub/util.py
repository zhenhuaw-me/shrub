import logging
from contextlib import contextmanager


def formatJSON(fname: str):
    """Format JSON for readability."""
    import json
    assert fname
    f = open(fname, 'r')
    j = json.load(f)
    f.close()

    f = open(fname, 'w')
    f.write(json.dumps(j, indent=2))
    f.close()


def args(index: int):
    """Get index-th arguments of the program"""
    import sys
    assert index >= 0 and index < len(sys.argv)
    return sys.argv[index]


# from
# https://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
@contextmanager
def suppressStdout():
    import sys
    import os
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


@contextmanager
def suppressLogging(level="error"):
    import logging
    logmap = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    new_level = logmap[level]
    logger = logging.getLogger()
    old_level = logger.getEffectiveLevel()
    logger.setLevel(new_level)
    try:
        yield
    finally:
        logger.setLevel(old_level)


def formatLogging(level):
    fmt = '%(asctime)s %(levelname).1s [%(name)s][%(filename)s:%(lineno)d] %(message)s'
    logging.basicConfig(format=fmt, level=level)
