from pathlib import Path
import csv
from logging import getLogger

logger = getLogger(__name__)

class Metrics(object):
    def __init__(self, writer, metrics_dir):
        self.loss = 0
        self.writer = writer
        self.metrics_dir = metrics_dir

    def logging(self, epoch, loss, mode):
        logger.info(f'{mode} metrics...')
        logger.info(f'loss: {loss}')

        # Change mode from 'test' to 'val' to change the display order from left to right to train and test.
        mode = 'val' if mode == 'test' else mode

        self.writer.add_scalar(f'loss/{mode}', loss, epoch)

    def save_csv(self, epoch, loss, mode):
        csv_path = self.metrics_dir / f'{mode}_metrics.csv'

        if not csv_path.exists():
            with open(csv_path, 'w') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(['epoch', f'{mode} loss'])

        with open(csv_path, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, loss])