#!/usr/bin/env python
from include import *
import argparse
from utils import get_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Analysis""",
                                     usage="%(prog)s ")
    parser.add_argument('--basedir', required=True)
    parser.add_argument('--analysis', nargs="+", required=True)
    args = parser.parse_known_args()[0]
    
    for analyse in args.analysis:
        aclass = get_class(analyse)
        a = aclass(args.basedir)
        a.update()
        print a.finish()
