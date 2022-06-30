from model.load_model import load_model


def run(word_list, save_path="saved_model_outputs/saved_models/pronouncer", model_type="standard"):
    """
    A runner for testing the transformer model out.
    """

    # Load the model
    transformer = load_model(save_path=save_path + "/" + model_type, model_type=model_type)

    # Make all words lower case - as transformer needs that
    word_list = [words.lower() for words in word_list]

    return transformer(word_list)


if __name__ == "__main__":
    print(run(["hello", "world"]))