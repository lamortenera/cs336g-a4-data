import fasttext

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", help="Path to the training data")
parser.add_argument("--test_path", help="Path to the test data")
parser.add_argument("--model_output", help="Path to the model output")
parser.add_argument("--args_json", help="JSON string containing extra arguments for fasttext training, e.g. '{\"lr\": 0.1, \"epoch\": 5}'", default="{}")


if __name__ == "__main__":
    args = parser.parse_args()
    train_args = json.loads(args.args_json)
    model = fasttext.train_supervised(args.train_path, **train_args)
    model.save_model(args.model_output)
    print("Model trained and saved to", args.model_output)
    num_test_samples, precision, recall = model.test(args.test_path)
    print(f"Test samples: {num_test_samples}, Precision: {precision:.4f}, Recall: {recall:.4f}")