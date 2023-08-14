from pathlib import Path

import pandas as pd
import pendulum
from openxes_cli.lib import csv_to_xes
from pix_framework.io.event_log import EventLogIDs


def convert_df_to_xes(df: pd.DataFrame, log_ids: EventLogIDs, output_path: Path):
    xes_datetime_format = "YYYY-MM-DDTHH:mm:ss.SSSZ"
    df[log_ids.start_time] = df[log_ids.start_time].apply(
        lambda x: pendulum.parse(x.isoformat()).format(xes_datetime_format)
    )
    df[log_ids.end_time] = df[log_ids.end_time].apply(
        lambda x: pendulum.parse(x.isoformat()).format(xes_datetime_format)
    )
    df.to_csv(output_path, index=False)
    csv_to_xes(output_path, output_path)
