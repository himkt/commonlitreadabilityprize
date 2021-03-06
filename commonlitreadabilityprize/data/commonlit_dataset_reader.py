from typing import Any, Dict, Iterable, MutableMapping, Optional
from urllib.parse import urlparse

from allennlp.data import DatasetReader
from allennlp.data import Tokenizer
from allennlp.data.fields.field import Field
from allennlp.data.fields import ArrayField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers.token_class import Token
import pandas
import numpy
from overrides import overrides


@DatasetReader.register("commonlit_reader")
class CommonlitDatasetReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer,
        excerpt_token_indexers: Optional[Dict[str, TokenIndexer]] = None,
        hostname_token_indexers: Optional[Dict[str, TokenIndexer]] = None,
    ) -> None:

        super().__init__()

        self.tokenizer = tokenizer
        self.excerpt_token_indexers: Dict[str, TokenIndexer] = excerpt_token_indexers or {
            "tokens": SingleIdTokenIndexer(),
        }
        self.hostname_token_indexers: Dict[str, TokenIndexer] = hostname_token_indexers or {
            "tokens": SingleIdTokenIndexer(),
        }

    def _read(self, file_path: str) -> Iterable[Instance]:
        instances = []

        dataframe = pandas.read_csv(file_path)
        dataframe["hostname"] = dataframe \
            .url_legal \
            .apply(lambda url: urlparse(url).hostname if isinstance(url, str) else "EMPTY_HOSTNAME")

        for _, row in dataframe.iterrows():
            excerpt = row.excerpt
            hostname = row.hostname
            target = row.target if hasattr(row, "target") else None
            instances.append(self.text_to_instance(excerpt, hostname, target))

        return instances

    @overrides
    def text_to_instance(self, excerpt: str, hostname: str, target: Optional[float] = None) -> Instance:
        excerpt_tokens = self.tokenizer.tokenize(excerpt)
        hostname_tokens = [Token(text=hostname)]
        fields: MutableMapping[str, Field[Any]] = {
            "excerpt": TextField(excerpt_tokens),
            "hostname": TextField(hostname_tokens),
        }
        if target is not None:
            fields["target"] = ArrayField(numpy.asarray(target, dtype=numpy.float32))
        return Instance(fields=fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        assert isinstance(instance.fields["excerpt"], TextField)
        instance.fields["excerpt"].token_indexers = self.excerpt_token_indexers
        assert isinstance(instance.fields["hostname"], TextField)
        instance.fields["hostname"].token_indexers = self.hostname_token_indexers
