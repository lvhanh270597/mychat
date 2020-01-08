import pickle
from sklearn.svm import LinearSVC
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

class Classifier:

    def __init__(self, dataset):
        self.__X, self.__y = dataset
        self.__vectorizer = TfidfVectorizer()
        self.__classifier = LinearSVC()

    def train(self):
        X = self.__vectorizer.fit_transform(self.__X).toarray()
        self.__classifier.fit(X, self.__y)
        pickle.dump(self.__classifier, open("cc.pkl", "wb"))
        print("before")
        print(self.__classifier.coef_)

    def load(self):
        print("after")
        self.__classifier = pickle.load(open("cc.pkl", "rb"))
        print(self.__classifier.coef_)

    def test(self):
        for x in self.__X:
            currentVector = self.__vectorizer.transform([x]).toarray()
            print("Sentence: %s, class: %s" % (x, self.__classifier.predict(currentVector)))
 

class Main:

    def __init__(self):
        X = """hey
xin cháo
hí
chào buổi sáng
chào buổi tối
tạm biệt
gặp lại bạn sau nhé
vâng
ừ
ok
tất nhiên rồi
đúng rồi
không
không bao giờ
tôi không nghĩ như vậy
đừng như vậy chứ
không còn cách nào
quá tuyệt vời
rất tốt
tốt
thật bất ngờ
xuất sắc
tôi cảm thấy rất tốt
buồn
rất là buồn
không vui
tệ hại
quá tệ
tệ quá
thật kinh khủng
không tốt lắm
bạn là robot à?
bạn là người à?
tôi đang nói chuyện với robot à?
hay tôi đang nói chuyện với người vậy?"""
        y = [
            "greet",
            "greet",
            "greet",
            "greet",
            "greet",

            "goodbye",
            "goodbye",
            
            "affirm",
            "affirm",
            "affirm",
            "affirm",
            "affirm",

            "deny",
            "deny",
            "deny",
            "deny",
            "deny",

            "mood_great",
            "mood_great",
            "mood_great",
            "mood_great",
            "mood_great",
            "mood_great",

            "mood_unhappy",
            "mood_unhappy",
            "mood_unhappy",
            "mood_unhappy",
            "mood_unhappy",
            "mood_unhappy",
            "mood_unhappy",
            "mood_unhappy",

            "bot_challenge",
            "bot_challenge",
            "bot_challenge",
            "bot_challenge",
        ]
        X = X.splitlines()
        listItems = [ViTokenizer.tokenize(xitem) for xitem in X]
        listLabels = y
        dataset = (listItems, listLabels)
        print(len(listItems), len(listLabels))
        clf = Classifier(dataset)
        # clf.train()
        clf.load()
        # clf.test()

if __name__ == "__main__":
    Main()