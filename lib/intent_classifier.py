import typing, os
from sklearn.neural_network import MLPClassifier
from typing import Any, Optional, Text, Dict
from rasa.nlu.components import Component
from rasa.nlu.training_data import TrainingData
from rasa.nlu import utils
from rasa.nlu.constants import (
    INTENT_ATTRIBUTE
)
if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


class IntentClassifier(Component):

    CLASSIFIER_PATH = "classifier.pkl"
    provides = []
    requires = []
    defaults = {}
    language_list = None

    def __init__(self, component_config=None):
        super().__init__(component_config)
        self.classifier = MLPClassifier(hidden_layer_sizes=(100,))

    def train(self, training_data : TrainingData, cfg, **kwargs):
        X, y = [], []
        for example in training_data.training_examples:
            X.append(example.get("vector"))
            y.append(example.get(INTENT_ATTRIBUTE))
        self.classifier.fit(X, y)
        for x, ypred in zip(training_data.training_examples, self.classifier.predict(X)):
            print(x.text, ypred)

    def process(self, message, **kwargs):
        currentVector = message.get("vector")
        intent = self.classifier.predict(currentVector)[0]
        message.set(
            "intent", self.convertToRasa(intent),
            add_to_output=True
        )

    def convertToRasa(self, output):
        return {
            "name": output
        }

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        classifier_file = os.path.join(model_dir, self.CLASSIFIER_PATH)
        utils.json_pickle(classifier_file, self)
        return {"classifier_file": self.CLASSIFIER_PATH}

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
        file_name = meta.get("classifier_file")
        classifier_file = os.path.join(model_dir, file_name)
        return utils.json_unpickle(classifier_file)
