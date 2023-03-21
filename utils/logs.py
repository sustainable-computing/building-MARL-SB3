from datetime import datetime
import os


def create_log_dir(log_dir, run_name):
    current_dt = datetime.now()
    dt_str = current_dt.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_dir, run_name, dt_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir
