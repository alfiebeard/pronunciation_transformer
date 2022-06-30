import sys
from model_evaluation.run import run
from speech_model.utils import join_list_ipa_audio, save_audio


if __name__ == "__main__":
    # Take in word or list of words
    if len(sys.argv) == 1:
        print("No arguments so demonstrating with I have a dream")
        word_list = ["i", "have", "a", "dream"]
    else:
        word_list = [word.lower() for word in sys.argv[1:]]
        
    # Run model
    predictions = run(word_list, save_path="saved_model_outputs/saved_models/pronouncer", model_type="standard")

    # Translate model outputs to audio files
    pronunciation_audio = join_list_ipa_audio(predictions)
    save_audio(pronunciation_audio, word_list=word_list)
