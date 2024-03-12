from ..logging.data_manager import DataManager


class VoidManager(DataManager):
    def __init__(self, config: dict):
        pass

    def update(self, log: dict, timestep: int):
        pass

    def log_images(self, name: str, images: list, timestep: int):
        pass

    def log_videos(self, name: str, video_frames: list, timestep: int):
        pass

    def save(self, agent, timestep: int):
        pass
