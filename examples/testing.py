from data.data_loader import load_data
from model_evaluation.test import evaluate_model

# Load test dataset in
(_, _, test_data, _, _) = load_data()

print('================ Standard Model ================')
_ = evaluate_model("saved_model_outputs/saved_models/pronouncer/standard", test_data)

print('================ Beam Model ================')
_ = evaluate_model("saved_model_outputs/saved_models/pronouncer/beam", test_data)