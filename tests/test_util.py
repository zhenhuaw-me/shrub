import logging
import shrub


def test_supress_stdout():
    print("Shall can see this print (1)")
    with shrub.util.suppressStdout():
        print("Shall NOT see this print")
    print("Shall can see this print (2)")


def test_supress_logging():
    logging.warning("Shall can see this warning (1)")
    with shrub.util.suppressLogging():
        logging.warning("Shall NOT see this warning")
    logging.warning("Shall can see this warning (2)")


if __name__ == '__main__':
    test_supress_stdout()
    test_supress_logging()
