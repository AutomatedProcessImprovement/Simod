from extraneous_activity_delays.config import SimulationModel
from lxml import etree

from simod.simulation.parameters.BPS_model import BPSModel


def make_simulation_model_from_bps_model(bps_model: BPSModel) -> SimulationModel:
    parser = etree.XMLParser(remove_blank_text=True)
    bpmn_model = etree.parse(bps_model.process_model, parser)
    parameters = bps_model.to_dict()

    simulation_model = SimulationModel(bpmn_model, parameters)

    return simulation_model
