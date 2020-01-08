from lib.structures.sentence import Sentence
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


class Extractor(Component):

    provides = []
    requires = []
    defaults = {}
    language_list = None

    def __init__(self, component_config=None):
        super().__init__(component_config)

    def train(self, training_data, cfg, **kwargs):
        for example in training_data.training_examples:
            sentence = Sentence(example.text)
            sentence.detect_name()
            example.set(
                "entities", sentence.get_extracted_names()
            )
        pass

    def process(self, message, **kwargs):
        sentence = Sentence(message.text)
        sentence.detect_name()
        message.set(
            "entities", sentence.get_extracted_names(),
            add_to_output=True
        )
    
    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        pass

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