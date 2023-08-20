"""This is a custom logger"""
import os
import sys
import logging
#logging format
LOGGINGSTR="[%(asctime)s:%(levelname)s:%(module)s:%(message)s]"
# Set up the log directory and file path
LOGDIR="logs"
log_filepath=os.path.join(LOGDIR,"running_logs.log")
os.makedirs(LOGDIR,exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=LOGGINGSTR,
   handlers= [
    logging.FileHandler(log_filepath),
    logging.StreamHandler(sys.stdout)
    ]
)
logger=logging.getLogger('ClassifierLogger')
