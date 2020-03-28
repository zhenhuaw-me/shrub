#!/usr/bin/env python

import logging
import shrub


def testSupressStdout():
    print("Shall can see this print (1)")
    with shrub.util.suppressStdout():
        print("Shall NOT see this print")
    print("Shall can see this print (2)")


def testSupressLogging():
    logging.warning("Shall can see this warning (1)")
    with shrub.util.suppressLogging():
        logging.warning("Shall NOT see this warning")
    logging.warning("Shall can see this warning (2)")


def main():
    testSupressStdout()
    testSupressLogging()


if __name__ == '__main__':
    main()
