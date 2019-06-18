import torch.nn as nn


CONFIG = 'config'
STATE = 'state'


class SavingModule(nn.Module):
    """
    SavingModule provides saving and loading functionality for a module.

    In order for a class to inherit from SavingModule, they need to:
    - have a class attribute config_cls with the class of the configuration to be used;
    - have an instance attribute called config storing this config.
    """
    config_cls = None

    def __init__(self):
        super().__init__()

        if self.config_cls is None:
            raise ValueError("Children classes need to set config_cls.")

    @classmethod
    def from_dict(cls, checkpoint: dict):
        assert CONFIG in checkpoint
        config = cls.config_cls(**checkpoint[CONFIG])
        module = cls(config)
        if STATE in checkpoint:
            module.load_state_dict(checkpoint[STATE])
        return module

    def to_dict(self, include_state=True):
        dic = {CONFIG: dict(self.config._asdict())}
        if include_state:
            dic[STATE] = self.state_dict()
        return dic
