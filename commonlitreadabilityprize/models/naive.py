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
from torch import cat
from torch import sqrt
from torch.nn import Linear


EPS = 1e-8


@Model.register("naive")
class NaiveRegressor(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        excerpt_embedder: TextFieldEmbedder,
        excerpt_encoder: Seq2VecEncoder,
        hostname_embedder: Optional[TextFieldEmbedder] = None,
    ) -> None:

        super().__init__(vocab)

        self.vocab = vocab
        self.excerpt_embedder = excerpt_embedder
        self.excerpt_encoder = excerpt_encoder
        self.hostname_embedder = hostname_embedder

        in_features = self.excerpt_encoder.get_output_dim()
        if hostname_embedder is not None:
            in_features += hostname_embedder.get_output_dim()

        self.classification_layer = Linear(
            in_features=in_features,
            out_features=1,
        )

    @overrides
    def forward(
        self,
        excerpt: TextFieldTensors,
        hostname: Optional[TextFieldTensors] = None,
        target: Optional[FloatTensor] = None,
    ) -> Dict[str, Tensor]:

        mask = get_text_field_mask(excerpt)
        excerpt_emb = self.excerpt_embedder(excerpt)
        hidden_state = self.excerpt_encoder(excerpt_emb, mask=mask)

        if self.hostname_embedder is not None and hostname is not None:
            hostname_emb = self.hostname_embedder(hostname)
            hidden_state = cat((hidden_state, hostname_emb.squeeze(dim=1)), dim=1)

        logit = self.classification_layer(hidden_state)

        output_dict = {"logit": logit}
        if target is not None:
            output_dict["loss"] = sqrt(mse_loss(logit.view(-1), target) + EPS)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {}
