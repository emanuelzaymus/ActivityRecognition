from src.classifiers import classifier
from src.classifiers.classifier import Preprocessing

WITH_PREVIOUS_CLASS_FEATURE = False

WINDOW_SIZES = [5, 7, 10, 12, 15, 17, 19, 22, 25, 27, 30, 32, 35, 37, 40]  # [5, 12, 19, 30, 40]

classifier.PREPROCESSOR = Preprocessing.STANDARD_SCALER
