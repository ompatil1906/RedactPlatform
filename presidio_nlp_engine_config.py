import logging
from typing import Tuple

from presidio_analyzer import RecognizerRegistry
from presidio_analyzer.nlp_engine import (
    NlpEngine,
    NlpEngineProvider,
)

logger = logging.getLogger("presidio-streamlit")



def create_nlp_engine_with_transformers(
    model_path: str,
) -> Tuple[NlpEngine, RecognizerRegistry]:
    """
    Instantiate an NlpEngine with a TransformersRecognizer and a small spaCy model.
    The TransformersRecognizer would return results from Transformers models, the spaCy model
    would return NlpArtifacts such as POS and lemmas.
    :param model_path: HuggingFace model path.
    """
    print(f"Loading Transformers model: {model_path} of type {type(model_path)}")

    nlp_configuration = {
        "nlp_engine_name": "transformers",
        "models": [
        {
            "lang_code": "en",
            "model_name": {"transformers": model_path, "spacy": "en_core_web_sm"},
        }
    ],
        "ner_model_configuration": {
            "model_to_presidio_entity_mapping": {
                "PER": "PERSON",
                "PERSON": "PERSON",
                "LOC": "LOCATION",
                "LOCATION": "LOCATION",
                "GPE": "LOCATION",
                "ORG": "ORGANIZATION",
                "ORGANIZATION": "ORGANIZATION",
                "NORP": "NRP",
                "AGE": "AGE",
                "ID": "ID",
                "EMAIL": "EMAIL",
                "PATIENT": "PERSON",
                "STAFF": "PERSON",
                "HOSP": "ORGANIZATION",
                "PATORG": "ORGANIZATION",
                "DATE": "DATE_TIME",
                "TIME": "DATE_TIME",
                "PHONE": "PHONE_NUMBER",
                "HCW": "PERSON",
                "HOSPITAL": "ORGANIZATION",
                "FACILITY": "LOCATION",
            },
            "low_confidence_score_multiplier": 0.4,
            "low_score_entity_names": ["ID"],
            "labels_to_ignore": [
                "CARDINAL",
                "EVENT",
                "LANGUAGE",
                "LAW",
                "MONEY",
                "ORDINAL",
                "PERCENT",
                "PRODUCT",
                "QUANTITY",
                "WORK_OF_ART",
            ],
        },
    }

    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)

    return nlp_engine, registry



def create_nlp_engine_with_azure_ai_language(ta_key: str, ta_endpoint: str):
    """
    Instantiate an NlpEngine with a TextAnalyticsWrapper and a small spaCy model.
    The TextAnalyticsWrapper would return results from calling Azure Text Analytics PII, the spaCy model
    would return NlpArtifacts such as POS and lemmas.
    :param ta_key: Azure Text Analytics key.
    :param ta_endpoint: Azure Text Analytics endpoint.
    """
    from azure_ai_language_wrapper import AzureAIServiceWrapper

    if not ta_key or not ta_endpoint:
        raise RuntimeError("Please fill in the Text Analytics endpoint details")

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()

    azure_ai_language_recognizer = AzureAIServiceWrapper(
        ta_endpoint=ta_endpoint, ta_key=ta_key
    )
    nlp_configuration = {
        "nlp_engine_name": "azure_ai_language",
        "models": [{"lang_code": "en", "model_name": None}],
    }

    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()

    registry.add_recognizer(azure_ai_language_recognizer)


    return nlp_engine, registry
