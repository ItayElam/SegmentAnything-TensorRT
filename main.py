import argparse
import numpy as np
import os
import cv2
from src.infer import InferenceEngine
from src.export import ExportSAM
from src.benchmark import PerformanceTester, AccuracyTester
from src.utils import choose_point


class NumpyArrayPoint(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, np.fromstring(values, sep=',').reshape(-1, 2))


class NumpyArrayLabel(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, np.fromstring(values, sep=',').reshape(-1))


def benchmark(sam_checkpoint, model_type, warmup_iters, measure_iters):
    valid_model_types = ['vit_b', 'vit_l', 'vit_h', 'all']
    assert (os.path.isfile(sam_checkpoint) or (os.path.isdir(
        sam_checkpoint) and model_type == "all")), f"Checkpoint file does not exist: {sam_checkpoint}"
    assert model_type in valid_model_types, f"Invalid model_type: {model_type}. It must be one of {valid_model_types}."

    if model_type == 'all':
        models = [os.path.join(sam_checkpoint, i) for i in os.listdir(sam_checkpoint) if "sam_vit_" in i]
        for m_type in ['vit_b', 'vit_l', 'vit_h']:
            for model in models:
                if m_type in os.path.split(model)[1]:
                    PerformanceTester(m_type, model).test_models(warmup_iters, measure_iters)
    else:
        PerformanceTester(model_type, sam_checkpoint).test_models(warmup_iters, measure_iters)


def accuracy(image_dir, model_type, sam_checkpoint, show_results, save_results):
    valid_model_types = ['vit_b', 'vit_l', 'vit_h', 'all']
    assert (os.path.isfile(sam_checkpoint) or (os.path.isdir(
        sam_checkpoint) and model_type == "all")), f"Checkpoint file does not exist: {sam_checkpoint}"
    assert model_type in valid_model_types, f"Invalid model_type: {model_type}. It must be one of {valid_model_types}."

    if model_type == 'all':
        models = [os.path.join(sam_checkpoint, i) for i in os.listdir(sam_checkpoint) if "sam_vit_" in i]
        for m_type in ['vit_b', 'vit_l', 'vit_h']:
            for model in models:
                if m_type in os.path.split(model)[1]:
                    AccuracyTester(image_dir, m_type, model, show_results, save_results).test_accuracy()
    else:
        AccuracyTester(image_dir, model_type, sam_checkpoint, show_results, save_results).test_accuracy()


def infer(pth_path, model_1, model_2, img_path):
    for file_path in [pth_path, model_1, img_path]:
        assert os.path.isfile(file_path), f"File does not exist: {file_path}"

    if model_2:
        assert os.path.isfile(model_2), f"File does not exist: {model_2}"
    input_point = choose_point(img_path)
    input_label = np.array([1])
    inference_engine = InferenceEngine(pth_path, model_1, model_2)
    image = cv2.imread(img_path)
    result = inference_engine(image, input_point, input_label)
    cv2.imshow("result", np.array(result).astype(np.uint8))
    cv2.waitKey(0)


def export(model_path, model_precision):
    assert os.path.isfile(model_path), f"Model file does not exist: {model_path}"
    sam_model = ExportSAM(model_path)
    sam_model(model_precision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparsers.required = True

    parser_benchmark = subparsers.add_parser('benchmark')
    parser_benchmark.add_argument('--sam_checkpoint', type=str, required=True)
    parser_benchmark.add_argument('--model_type', type=str, required=True)
    parser_benchmark.add_argument('--warmup_iters', type=int, default=5)
    parser_benchmark.add_argument('--measure_iters', type=int, default=50)
    parser_benchmark.set_defaults(func=benchmark)

    parser_infer = subparsers.add_parser('infer')
    parser_infer.add_argument('--pth_path', type=str, required=True)
    parser_infer.add_argument('--model_1', type=str, required=True)
    parser_infer.add_argument('--model_2', type=str, default=None)
    parser_infer.add_argument('--img_path', type=str, required=True)
    parser_infer.set_defaults(func=infer)

    parser_export = subparsers.add_parser('export')
    parser_export.add_argument('--model_path', type=str, required=True)
    parser_export.add_argument('--model_precision', type=str, choices=['fp32', 'fp16', 'both'], required=True)
    parser_export.set_defaults(func=export)

    parser_accuracy = subparsers.add_parser('accuracy')
    parser_accuracy.add_argument('--image_dir', type=str, required=True)
    parser_accuracy.add_argument('--model_type', type=str, required=True)
    parser_accuracy.add_argument('--sam_checkpoint', type=str, required=True)
    parser_accuracy.add_argument('--show_results', action='store_true', default=False)
    parser_accuracy.add_argument('--save_results', action='store_true', default=False)
    parser_accuracy.set_defaults(func=accuracy)

    args = parser.parse_args()
    func = vars(args).pop('func')
    func(**vars(args))
