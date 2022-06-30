import sys
from model_evaluation.run import run

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No arguments so demonstrating with hello world")
        word_list = ["hello", "world"]
    else:
        word_list = [word.lower() for word in sys.argv[1:]]
        
    predictions = run(word_list, save_path="saved_model_outputs/saved_models/pronouncer", model_type="standard")
    for word, prediction in zip(word_list, predictions):
        print(word + " -> " + prediction)