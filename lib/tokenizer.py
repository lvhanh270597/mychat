from pyvi import ViTokenizer
from rasa.nlu.components import Component
import typing
from typing import Any, Optional, Text, Dict
from rasa.nlu.constants import (
    INTENT_ATTRIBUTE,
    TEXT_ATTRIBUTE,
    TOKENS_NAMES,
    MESSAGE_ATTRIBUTES,
)
if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


class Tokenizer(Component):

    provides = []
    requires = []
    defaults = {}
    language_list = None

    def __init__(self, component_config=None):
        super().__init__(component_config)

    def train(self, training_data, cfg, **kwargs):
        for example in training_data.training_examples:
            textToken = self.tokenize(example.text)
            example.set(TEXT_ATTRIBUTE, textToken)
            example.text = textToken

    def process(self, message, **kwargs):
        textToken = self.tokenize(message.text)
        message.set(TEXT_ATTRIBUTE, textToken)
        message.text = textToken

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        pass

    def tokenize(self, text):
        return ViTokenizer.tokenize(text)

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any,
    ) -> "Component":
        """Load this component from file."""

        if cached_component:
            return cached_component
        else:
            return cls(meta)