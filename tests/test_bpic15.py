import pytest
from simod.event_log.event_log import EventLog
from simod.settings.simod_settings import SimodSettings
from simod.simod import Simod


@pytest.mark.system
def test_bpic15(entry_point):
    settings = SimodSettings.from_path(entry_point / "bpic15/bpic15_1_with_model_v4.yml")

    event_log = EventLog.from_path(
        train_log_path=settings.common.train_log_path,
        log_ids=settings.common.log_ids,
        preprocessing_settings=settings.preprocessing,
    )
    optimizer = Simod(settings, event_log=event_log)
    optimizer.run()

    assert optimizer.final_bps_model.resource_model is not None
