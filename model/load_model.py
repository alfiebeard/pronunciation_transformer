from model.pronouncer import LoadPronouncer
from data.embeddings.embedding import load_embedding


def load_model(save_path="saved_model_outputs/saved_models/pronouncer/", embedding_path="data/embeddings/saved_embeddings/", model_type="standard"):
    """
    Load a previously saved model and embeddings
    """
    word_embedding = load_embedding(embedding_path + "word_embedding")
    ipa_embedding = load_embedding(embedding_path + "ipa_embedding")
    reloaded = LoadPronouncer(save_path, word_embedding, ipa_embedding, model_type=model_type)
    return reloaded
