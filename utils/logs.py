from datetime import datetime
import os


def create_log_dir(log_dir, run_name="", use_dt_str=True):
    if use_dt_str:
        current_dt = datetime.now()
        dt_str = current_dt.strftime("%Y-%m-%d_%H-%M-%S")
        if run_name != "":
            log_dir = os.path.join(log_dir, run_name, dt_str)
        else:
            log_dir = os.path.join(log_dir, dt_str)
    else:
        if run_name != "":
            log_dir = os.path.join(log_dir, run_name)
        else:
            log_dir = os.path.join(log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir
