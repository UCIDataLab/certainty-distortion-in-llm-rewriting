import numpy as np

from functools import partial
## Uncertainty
from certainty_estimator.predict_certainty import CertaintyEstimator 
from nltk import sent_tokenize, word_tokenize
from metrics import score_uncertainty, stable_softmax, entropy


print("Initializing the aspect-level uncertainty estimator from Pei and Jurgen (2021)")
UNCERTAINTY_MODEL_ASPECT = CertaintyEstimator("aspect-level", cuda=True)


def predict_pei_jurgens(text, how="mean", unc_model=UNCERTAINTY_MODEL_ASPECT):
    # Break the text into the individual sentences
    sents = sent_tokenize(text)
    sents = [s for s in sents if len(s.strip()) > 0]

    # Average the classifiers' predictions over all sentences
    preds_per_sent = unc_model.predict(sents, get_processed_output=False)[0]
    preds_per_sent = [stable_softmax(preds) for preds in preds_per_sent]

    if how in ("mean", "avg"):
        preds = sum(preds_per_sent) / len(preds_per_sent)
    elif how in ("last",):
        preds = preds_per_sent[-1]
    elif how in ("first"): 
        preds = preds_per_sent[0]
    elif how in ("all",):
        preds = unc_model.predict(text, get_processed_output=False)[0][0] #? 
        preds = stable_softmax(preds)
    else:
        raise ValueError(f"Unexpected `how`: {how}")
    return preds


def predict_category(scores, category):
    cat_id = CertaintyEstimator.UNCERTAINTY_CATEGORIES.index(category)
    # Note: cat_id will err if category not in the list
    preds = scores[cat_id].reshape(1, -1)
    # print(preds.shape) (1,3)
    return preds[0] # returns array of shape (3,) 
        

def predict_categories(scores, categories=["Framing", "Probability"]):
    cat_id = [CertaintyEstimator.UNCERTAINTY_CATEGORIES.index(category) for category in categories]
    if len(cat_id) == 0:
        cat_id = np.arange(len(CertaintyEstimator.UNCERTAINTY_CATEGORIES))

    preds = scores[cat_id, :]
    return preds.sum(axis=0) / len(categories)


def predict_max_certain_categories(scores, categories=["Framing", "Probability", "Number", "Extent"]):
    cat_id = [CertaintyEstimator.UNCERTAINTY_CATEGORIES.index(category) for category in categories]
    if len(cat_id) == 0:
        cat_id = np.arange(len(CertaintyEstimator.UNCERTAINTY_CATEGORIES))

    # {0: 'Uncertain', 1: 'NotPresent', 2: 'Certain'}
    preds = scores[cat_id,2]
    return preds.max()
    

def predict_max_uncertain_categories(scores, categories=["Framing", "Probability", "Number", "Extent"]):
    cat_id = [CertaintyEstimator.UNCERTAINTY_CATEGORIES.index(category) for category in categories]
    if len(cat_id) == 0:
        cat_id = np.arange(len(CertaintyEstimator.UNCERTAINTY_CATEGORIES))

    # {0: 'Uncertain', 1: 'NotPresent', 2: 'Certain'}
    preds = scores[cat_id, 0]
    return preds.max()


def predict_not_present_categories(scores, categories=["Framing", "Probability", "Number", "Extent"]):
    cat_id = [CertaintyEstimator.UNCERTAINTY_CATEGORIES.index(category) for category in categories]
    if len(cat_id) == 0:
        cat_id = np.arange(len(CertaintyEstimator.UNCERTAINTY_CATEGORIES))
    preds = scores
    return min(preds[:,1])




predict_framing = partial(predict_category, category="Framing")
predict_extent = partial(predict_category, category="Extent")
predict_number = partial(predict_category, category="Number")
predict_probability = partial(predict_category, category="Probability")

