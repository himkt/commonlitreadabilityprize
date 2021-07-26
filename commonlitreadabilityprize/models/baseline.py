from typing import Dict, Optional
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules import Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.data.fields.text_field import TextFieldTensors
from overrides.overrides import overrides
from torch import FloatTensor
from torch.functional import Tensor
from torch.nn.functional import mse_loss
from torch import sqrt
from torch.nn import Linear


EPS = 1e-8


@Model.register("baseline")
class BaselineRegressor(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        excerpt_embedder: TextFieldEmbedder,
        excerpt_encoder: Seq2VecEncoder,
    ) -> None:

        super().__init__(vocab)

        self.vocab = vocab
        self.excerpt_embedder = excerpt_embedder
        self.excerpt_encoder = excerpt_encoder

        self.dense = Linear(
            in_features=self.excerpt_encoder.get_output_dim(),
            out_features=1,
        )

    @overrides
    def forward(
        self,
        excerpt: TextFieldTensors,
        target: Optional[FloatTensor] = None,
    ) -> Dict[str, Tensor]:

        mask = get_text_field_mask(excerpt)
        excerpt_emb = self.excerpt_embedder(excerpt)
        logit = self.dense(self.excerpt_encoder(excerpt_emb, mask=mask))

        output_dict = {"logit": logit}
        if target is not None:
            output_dict["loss"] = sqrt(mse_loss(logit.view(-1), target) + EPS)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {}
