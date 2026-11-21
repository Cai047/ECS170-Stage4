"""Stage 4 classification entry point in the instructor-provided project structure."""
import argparse
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN import Method_RNN
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_4_code.Setting_Train_Validation_Test import Setting_Train_Validation_Test
from local_code.stage_4_code.experiment import predict

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='stage_4_data.zip or text_classification folder')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'result' / 'stage_4_result')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--vocab-size', type=int, default=30000)
    parser.add_argument('--device', choices=['auto', 'cpu'], default='auto')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--text')
    args = parser.parse_args()
    if args.checkpoint:
        if args.text is None: parser.error('--text is required for prediction')
        predict(args)
    else:
        if args.data is None: parser.error('--data is required for training')
        data_obj = Dataset_Loader('IMDb', 'Official train/test partitions')
        data_obj.dataset_source_folder_path = str(args.data)
        method_obj = Method_RNN('RNN/LSTM/GRU', 'Controlled cell and hidden-size comparisons', args)
        result_obj = Result_Saver('Stage 4 results', 'JSON summary and checkpoint artifacts')
        result_obj.result_destination_folder_path = str(args.output)
        result_obj.result_destination_file_name = 'results.json'
        evaluate_obj = Evaluate_Metrics('Classification metrics', 'Positive precision/recall/F1 and macro-F1')
        setting_obj = Setting_Train_Validation_Test('Train/validation/test', 'Seeded 80/20 stratified training split')
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        setting_obj.print_setup_summary()
        scores, _ = setting_obj.load_run_save_evaluate()
        print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
