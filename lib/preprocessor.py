import typing, pickle
from vicorrect.model import CorrectVietnameseSentence
from typing import Any, Optional, Text, Dict
from rasa.nlu.components import Component
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import (
    INTENT_ATTRIBUTE,
    TEXT_ATTRIBUTE,
    TOKENS_NAMES,
    MESSAGE_ATTRIBUTES,
)
if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


class Preprocessor(Component):

    # provides = []
    provides = [TOKENS_NAMES[attribute] for attribute in MESSAGE_ATTRIBUTES]
    requires = []
    defaults = {}
    language_list = None
    CORRECT_PATH = "data/corrector/corrector.pkl"

    def __init__(self, component_config=None):
        super().__init__(component_config)
        self.__loadCorrector()

    def __loadCorrector(self):
        try:
            with open(self.CORRECT_PATH, "rb") as fp:
                self.__corrector = pickle.load(fp)
        except Exception as E:
            print("Error: %s" % E)
            self.__corrector = CorrectVietnameseSentence()

    def train(self, training_data : TrainingData, cfg, **kwargs):
        for example in training_data.training_examples:
            sentences = self.__corrector.predict([example.text])[0]
            sentence = example.text if len(sentences) == 0 else sentences[0]
            example.text = sentence
            example.set(TEXT_ATTRIBUTE, sentence)

    def process(self, message, **kwargs):
        sentences = self.__corrector.predict([message.text])[0]
        sentence = message.text if len(sentences) == 0 else sentences[0]
        message.text = sentence
        message.set(TEXT_ATTRIBUTE, sentence)

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