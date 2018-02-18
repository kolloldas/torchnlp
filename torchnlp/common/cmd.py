import sys
import re

def run_cmd(default=None, **kwargs):
    """
    Simple command runner
    """
    if len(sys.argv) > 1:
        m = re.match(r'^--(.+)', sys.argv[1])
        if m and m.group(1) in kwargs:
            kwargs[m.group(1)]()
    elif default is not None:
        default()