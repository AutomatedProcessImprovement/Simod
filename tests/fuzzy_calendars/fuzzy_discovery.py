import os

from datetime import datetime, timedelta
import pandas as pd

from scipy.spatial import distance
import numpy as np

from simod.fuzzy_calendars.fuzzy_discovery import build_fuzzy_calendars

experiment_logs = {
    0: "loan_SC_LU",
    1: "loan_SC_HU",
    2: "loan_MC_LU",
    3: "loan_MC_HU",
}

process_files = {
    "loan_SC_LU": {
        "xes_log": "./../assets/fuzzy/in/xes_logs/loan_SC_LU.xes",
        "csv_log": "./../assets/fuzzy/in/csv_logs/loan_SC_LU.csv",
        "bpmn_model": "./../assets/fuzzy/in/bpmn_models/LoanOriginationModel.bpmn",
        "json": "./../assets/fuzzy/out/json/loan_SC_LU.json",
        "sim_log": "./../assets/fuzzy/out/prosimos/logs/loan_SC_LU_log.csv",
        "sim_stats": "./../assets/fuzzy/out/prosimos/stats/loan_SC_LU_stat.csv",
        "start_datetime": "2015-03-06 15:47:26+00:00",
        "total_cases": 1000,
    },
    "loan_SC_HU": {
        "xes_log": "./../assets/fuzzy/in/xes_logs/loan_SC_HU.xes",
        "csv_log": "./../assets/fuzzy/in/csv_logs/loan_SC_HU.csv",
        "bpmn_model": "./../assets/fuzzy/in/bpmn_models/LoanOriginationModel.bpmn",
        "json": "./../assets/fuzzy/out/json/loan_SC_HU.json",
        "sim_log": "./../assets/fuzzy/out/prosimos/logs/loan_SC_HU.csv",
        "sim_stats": "./../assets/fuzzy/out/prosimos/stats/loan_SC_HU.csv",
        "start_datetime": "2015-03-06 15:47:26+00:00",
        "total_cases": 1000,
    },
    "loan_MC_LU": {
        "xes_log": "./../assets/fuzzy/in/xes_logs/loan_MC_LU.xes",
        "csv_log": "./../assets/fuzzy/in/csv_logs/loan_MC_LU.csv",
        "bpmn_model": "./../assets/fuzzy/in/bpmn_models/LoanOriginationModel.bpmn",
        "json": "./../assets/fuzzy/out/json/loan_MC_LU.json",
        "sim_log": "./../assets/fuzzy/out/prosimos/logs/loan_MC_LU.csv",
        "sim_stats": "./../assets/fuzzy/out/prosimos/stats/loan_MC_LU.csv",
        "start_datetime": "2015-03-09 09:00:26+00:00",
        "total_cases": 1000,
    },
    "loan_MC_HU": {
        "xes_log": "./../assets/fuzzy/in/xes_logs/loan_MC_HU.xes",
        "csv_log": "./../assets/fuzzy/in/csv_logs/loan_MC_HU.csv",
        "bpmn_model": "./../assets/fuzzy/in/bpmn_models/LoanOriginationModel.bpmn",
        "json": "./../assets/fuzzy/out/json/loan_MC_HU.json",
        "sim_log": "./../assets/fuzzy/out/prosimos/logs/loan_MC_HU.csv",
        "sim_stats": "./../assets/fuzzy/out/prosimos/stats/loan_MC_HU.csv",
        "start_datetime": "2015-03-06 15:47:26+00:00",
        "total_cases": 1000,
    },
}


def main():
    for model_name in process_files:
        build_fuzzy_calendars(
            process_files[model_name]["csv_log"],
            process_files[model_name]["bpmn_model"],
            process_files[model_name]["json"],
            15,
        )
        break
    os._exit(0)


if __name__ == "__main__":
    main()
