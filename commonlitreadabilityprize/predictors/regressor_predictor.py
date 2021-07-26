from allennlp.common.util import JsonDict
from allennlp.data.instance import Instance
from allennlp.predictors import Predictor


@Predictor.register("regressor_predictor")
class RegressorPredictor(Predictor):
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(**json_dict)  # type: ignore
