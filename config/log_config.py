import sys
import logging

logging.basicConfig(level=logging.INFO, \
                    format='%(asctime)s [%(levelname)s][%(module)s - (funcName)s] %(message)s', \
                    datefmt='%a, %d %b %Y %H:%M:%S', stream=sys.stdout)