import json
import timeit


class RuntimeMeter:

    runtime_start: dict
    runtime_stop: dict
    runtimes: dict

    TOTAL: str = "SIMOD_TOTAL_RUNTIME"
    PREPROCESSING: str = "preprocessing"
    INITIAL_MODEL: str = "discover-initial-BPS-model"
    CONTROL_FLOW_MODEL: str = "optimize-control-flow-model"
    RESOURCE_MODEL: str = "optimize-resource-model"
    DATA_ATTRIBUTES_MODEL: str = "discover-data-attributes"
    EXTRANEOUS_DELAYS: str = "discover-extraneous-delays"
    FINAL_MODEL: str = "discover-final-BPS-model"
    EVALUATION: str = "evaluate-final-BPS-model"

    def __init__(self):
        self.runtime_start = dict()
        self.runtime_stop = dict()
        self.runtimes = dict()

    def start(self, stage_name: str):
        self.runtime_start[stage_name] = timeit.default_timer()

    def stop(self, stage_name: str):
        self.runtime_stop[stage_name] = timeit.default_timer()
        self.runtimes[stage_name] = self.runtime_stop[stage_name] - self.runtime_start[stage_name]

    def to_json(self) -> str:
        return json.dumps(self.runtimes)
