from ..logging.data_manager import DataManager


class VoidManager(DataManager):
    def __init__(self, config: dict):
        pass

    def update(self, log: dict, timestep: int):
        pass

    def save(self, agent, timestep: int):
        pass
