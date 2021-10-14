from transformers.models.bert.configuration_bert import BertConfig

"""
This file contains Configs for temporal models.
Each Config class inherits from its non-temporal Config and adds several temporal members.

    times (:obj:`List[int]`, `optional`, defaults to `None`):
        List of time points.
    time_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"prepend_token"`):
        Type of time embedding. Currently supported: :obj:`"prepend_token"`, :obj:`"prepend_nl_token"`.
"""


class TempoBertConfig(BertConfig):
    model_type = "tempobert"

    def __init__(self, times=None, time_embedding_type="prepend_token", **kwargs):
        super().__init__(**kwargs)
        self.init_tempo_config(times, time_embedding_type)
