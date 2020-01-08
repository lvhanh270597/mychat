import typing, os
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Any, Optional, Text, Dict
from rasa.nlu.components import Component
from rasa.nlu.training_data import TrainingData
from rasa.nlu import utils
from rasa.nlu.constants import (
    INTENT_ATTRIBUTE,
    TEXT_ATTRIBUTE,
)
if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


class Vectorizer(Component):

    STOPWORDS_PATH = "data/stopwords/vietnamese-stopwords-dash.txt"
    VECTOR_PATH = "vectorizer.pkl"
    provides = []
    requires = []
    defaults = {}
    language_list = None

    def __init__(self, component_config=None):
        super().__init__(component_config)
        # self.__readStopwords()
        self.__vectorizer = TfidfVectorizer(
            # stop_words=self.__stopwords
        )

    def train(self, training_data : TrainingData, cfg, **kwargs):
        dataTokens = []
        for example in training_data.training_examples:
            dataTokens.append(example.get(TEXT_ATTRIBUTE))
        dataVectors = self.__vectorizer.fit_transform(dataTokens)
        dataVectors = dataVectors.toarray()
        for i, example in enumerate(training_data.training_examples):
            example.set("vector", dataVectors[i])

    def __readStopwords(self):
        try:
            with open(self.STOPWORDS_PATH) as fp:
                self.__stopwords = set(fp.read().splitlines())
        except Exception as E:
            print("Error: %s" % E)
            self.__stopwords = set()

    def process(self, message, **kwargs):
        currentVector = self.__vectorizer.transform([message.get(TEXT_ATTRIBUTE)])
        currentVector = currentVector.toarray()
        message.set("vector", currentVector)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        vectorizer_file = os.path.join(model_dir, self.VECTOR_PATH)
        utils.json_pickle(vectorizer_file, self)
        return {"vectorizer_file": self.VECTOR_PATH}    

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
        file_name = meta.get("vectorizer_file")
        vectorizer_file = os.path.join(model_dir, file_name)
        return utils.json_unpickle(vectorizer_file)
