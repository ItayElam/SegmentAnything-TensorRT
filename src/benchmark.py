import datetime
import random
import tensorrt as trt
from prettytable import PrettyTable
from segment_anything import sam_model_registry, SamPredictor
import torch
import onnxruntime as ort
import os
import timeit
import numpy as np
from src.infer import InferSamH, InferSamBL
import src.utils as utils
import cv2

screen_width = 1600
screen_height = 900


def calculate_iou(prediction, ground_truth):
    # Calculate Intersection over Union (IOU) score
    intersection = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


class DefaultModel:
    def __init__(self, pth_path, model_type):
        sam = sam_model_registry[model_type](checkpoint=pth_path)
        self.predictor = SamPredictor(sam.to('cuda'))

    def infer(self, input_image, input_point, input_label):
        self.predictor.set_image(input_image)
        masks, scores, logits = self.predictor.predict(point_coords=input_point, point_labels=input_label,
                                                       multimask_output=False)

        return masks


class AccuracyTester:
    def __init__(self, img_dir, model_type, sam_path, show_results, save_results):
        self.infer32 = None
        self.default_model = None
        self.table = None
        self.infer16 = None
        self.img_dir = img_dir
        self.model_type = model_type
        self.sam_path = sam_path
        self.show_results = show_results
        self.save_results = save_results

        if save_results:
            current_datetime = datetime.datetime.now()
            datetime_string = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
            dir_name = f"accuracy_test/{model_type}/{datetime_string}"
            self.output_dir = dir_name
            os.makedirs(self.output_dir)

    def _get_infer_obj(self, precision):
        if self.model_type == 'vit_h':
            trt_model_path = f'exported_models/{self.model_type}/model_{precision}_1.engine'
            trt_model_path_2 = f'exported_models/{self.model_type}/model_{precision}_2.engine'
            return InferSamH(self.sam_path, trt_model_path, trt_model_path_2, self.model_type)
        else:
            trt_model_path = f'exported_models/{self.model_type}/model_{precision}.engine'

            return InferSamBL(self.sam_path, trt_model_path, self.model_type)

    def _get_img_paths(self):
        return [os.path.join(self.img_dir, img_name) for img_name in os.listdir(self.img_dir)]

    def load_image_and_data(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        points = []
        labels = []
        for i in range(random.randint(1, 6)):
            points.append((random.randint(0, image.shape[1]), random.randint(0, image.shape[0])))
            labels.append(1)
        return image, np.array(points), np.array(labels)

    def test_accuracy(self):
        self.table = PrettyTable()
        self.table.field_names = ["Model", "Minimum IOU", "Mean IOU"]
        self.table.title = f"IOU Comparison for {self.model_type}"

        img_paths = self._get_img_paths()
        iou_scores_32 = []
        iou_scores_16 = []

        data_arr = []
        for img_path in img_paths:
            data_arr.append(self.load_image_and_data(img_path))
        output_image_default_arr = []
        output_image_fp32_arr = []
        output_image_fp16_arr = []
        self.default_model = DefaultModel(self.sam_path, self.model_type)
        for input_image, input_point, input_label in data_arr:
            output_image_default = self.default_model.infer(input_image, input_point, input_label)
            output_image_default_arr.append(output_image_default)
        del self.default_model
        torch.cuda.empty_cache()
        self.infer32 = self._get_infer_obj('fp32')
        for input_image, input_point, input_label in data_arr:
            output_image32 = self.infer32.infer(input_image, input_point, input_label)
            output_image_fp32_arr.append(output_image32)
        del self.infer32
        torch.cuda.empty_cache()
        self.infer16 = self._get_infer_obj('fp16')
        for input_image, input_point, input_label in data_arr:
            output_image16 = self.infer16.infer(input_image, input_point, input_label)
            output_image_fp16_arr.append(output_image16)
        del self.infer16
        torch.cuda.empty_cache()
        image_count = 0
        for output_image_default, output_image32, output_image16, (original_image, input_point, _) in zip(
                output_image_default_arr, output_image_fp32_arr, output_image_fp16_arr, data_arr):
            iou_score_32 = calculate_iou(output_image32, output_image_default)
            iou_score_16 = calculate_iou(output_image16, output_image_default)
            if self.show_results or self.save_results:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                output_image_default = np.transpose(output_image_default * 255, (1, 2, 0)).astype(np.uint8)
                output_image32 = np.transpose(output_image32 * 255, (1, 2, 0)).astype(np.uint8)
                output_image16 = np.transpose(output_image16 * 255, (1, 2, 0)).astype(np.uint8)
                output_image_default = cv2.cvtColor(output_image_default, cv2.COLOR_GRAY2RGB)
                output_image32 = cv2.cvtColor(output_image32, cv2.COLOR_GRAY2RGB)
                output_image16 = cv2.cvtColor(output_image16, cv2.COLOR_GRAY2RGB)

                for point in input_point:
                    cv2.circle(original_image, point, 3 * (max(original_image.shape) // 800), (0, 0, 255), -1)
                    cv2.circle(output_image_default, point, 3 * (max(original_image.shape) // 800), (0, 0, 255), -1)
                    cv2.circle(output_image32, point, 3 * (max(original_image.shape) // 800), (0, 0, 255), -1)
                    cv2.circle(output_image16, point, 3 * (max(original_image.shape) // 800), (0, 0, 255), -1)

                original_image = cv2.resize(original_image, (screen_width // 2, screen_height // 2))
                output_image_default = cv2.resize(output_image_default, (screen_width // 2, screen_height // 2))
                output_image32 = cv2.resize(output_image32, (screen_width // 2, screen_height // 2))
                output_image16 = cv2.resize(output_image16, (screen_width // 2, screen_height // 2))

                screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

                screen[:screen_height // 2, :screen_width // 2] = original_image
                screen[:screen_height // 2, screen_width // 2:] = output_image_default
                screen[screen_height // 2:, :screen_width // 2] = output_image32
                screen[screen_height // 2:, screen_width // 2:] = output_image16

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(screen, 'Original', (10, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(screen[:screen_height // 2, screen_width // 2:], 'PyTorch', (10, 20), font, 0.5,
                            (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(screen[screen_height // 2:, :screen_width // 2], 'TensorRT_FP32', (10, 20), font, 0.5,
                            (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(screen[screen_height // 2:, screen_width // 2:], 'TensorRT_FP16', (10, 20), font, 0.5,
                            (0, 255, 0), 1, cv2.LINE_AA)

                if self.show_results:
                    cv2.imshow("Combined Images", screen)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                elif self.save_results:
                    cv2.imwrite(os.path.join(self.output_dir, f"{image_count}.jpg"), screen)
                    image_count += 1
            iou_scores_32.append(iou_score_32)
            iou_scores_16.append(iou_score_16)

        min_iou_score_32 = np.min(iou_scores_32)
        mean_iou_score_32 = np.mean(iou_scores_32)
        min_iou_score_16 = np.min(iou_scores_16)
        mean_iou_score_16 = np.mean(iou_scores_16)

        self.table.add_row([f"{self.model_type} FP32", round(min_iou_score_32, 4), round(mean_iou_score_32, 4)])
        self.table.add_row([f"{self.model_type} FP16", round(min_iou_score_16, 4), round(mean_iou_score_16, 4)])

        print(self.table)


class PerformanceTester:
    def __init__(self, model_type, sam_checkpoint, image=None):
        self.model_type = model_type
        self.sam_checkpoint = sam_checkpoint
        self.device = "cuda"
        if image is None:
            self.input_image = np.random.randint(0, 256, (1080, 1920, 3), dtype='uint8')
        else:
            self.input_image = image
        self.sam = None
        self.sam_image_encoder = None
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53])
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375])
        self.img_size = 1024
        self.input_image_processed = None
        self.ort_session = None
        self.runtime = trt.Runtime(trt.Logger(min_severity=trt.ILogger.ERROR))
        self.context = None
        self.context_fp16 = None
        self.performance_table = PrettyTable()
        self.performance_table.field_names = ["Model", "Average FPS", "Average Time (sec)", "Relative FPS",
                                              "Relative Time (%)"]
        self.performance_table.title = f"Performance Comparison for {self.model_type}"

    def add_to_table(self, name, fps, times, base_fps=None, base_times=None):
        avg_fps = round(fps, 2)
        avg_time = round(times, 6)
        if base_fps and base_times:
            rel_fps = round(fps / base_fps, 2)
            rel_time = round(times / base_times * 100, 2)
            self.performance_table.add_row([name, avg_fps, avg_time, rel_fps, rel_time])
        else:
            self.performance_table.add_row([name, avg_fps, avg_time, 'NA', 'NA'])

    def print_stats(self):
        print(self.performance_table)

    def load_pytorch_model(self):
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam_image_encoder = SamPredictor(self.sam.to(self.device))
        self.input_image_processed = utils.preprocess_image(self.input_image, 1024, self.device, self.pixel_mean,
                                                            self.pixel_std, self.img_size)

    def unload_pytorch_model(self):
        del self.sam
        del self.pixel_mean
        del self.pixel_std
        del self.img_size
        del self.sam_image_encoder
        self.input_image_processed = self.input_image_processed.cpu()
        torch.cuda.empty_cache()

    def load_tensorrt_model_fp32(self):
        engine_path = f"exported_models/{self.model_type}/model_fp32.engine"
        if not os.path.isfile(engine_path):
            raise FileNotFoundError("Model has not been generated yet, build the model first.")
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        self.context = self.runtime.deserialize_cuda_engine(serialized_engine).create_execution_context()

    def unload_tensorrt_model_fp32(self):
        self.context = None
        torch.cuda.empty_cache()

    def load_tensorrt_model_fp16(self):
        engine_path_fp16 = f"exported_models/{self.model_type}/model_fp16.engine"
        if not os.path.isfile(engine_path_fp16):
            raise FileNotFoundError(f"Model has not been generated yet, build the model first.\n{engine_path_fp16}")
        with open(engine_path_fp16, "rb") as f:
            serialized_engine_fp16 = f.read()
        self.context_fp16 = self.runtime.deserialize_cuda_engine(serialized_engine_fp16).create_execution_context()

    def unload_tensorrt_model_fp16(self):
        self.context_fp16 = None
        torch.cuda.empty_cache()

    def load_tensorrt_model_ensemble(self):
        TRT_LOGGER = trt.Logger()
        runtime = trt.Runtime(TRT_LOGGER)
        engine_path1 = f"exported_models/{self.model_type}/model_fp32_1.engine"
        if not os.path.isfile(engine_path1):
            raise FileNotFoundError("model has not been generated yet, build model first")
        with open(engine_path1, "rb") as f:
            serialized_engine1 = f.read()
        engine1 = runtime.deserialize_cuda_engine(serialized_engine1)
        context1 = engine1.create_execution_context()

        engine_path2 = f"exported_models/{self.model_type}/model_fp32_2.engine"
        if not os.path.isfile(engine_path2):
            raise FileNotFoundError("model has not been generated yet, build model first")
        with open(engine_path2, "rb") as f:
            serialized_engine2 = f.read()
        engine2 = runtime.deserialize_cuda_engine(serialized_engine2)
        context2 = engine2.create_execution_context()
        self.context = [context1, context2]

    def unload_tensorrt_model_ensemble(self):
        self.context = None
        torch.cuda.empty_cache()

    def load_tensorrt_model_fp16_ensemble(self):
        TRT_LOGGER = trt.Logger()
        runtime = trt.Runtime(TRT_LOGGER)
        engine_path1 = f"exported_models/{self.model_type}/model_fp16_1.engine"
        if not os.path.isfile(engine_path1):
            raise FileNotFoundError("model has not been generated yet, build model first")
        with open(engine_path1, "rb") as f:
            serialized_engine1 = f.read()
        engine1 = runtime.deserialize_cuda_engine(serialized_engine1)
        context1 = engine1.create_execution_context()

        engine_path2 = f"exported_models/{self.model_type}/model_fp16_2.engine"
        if not os.path.isfile(engine_path2):
            raise FileNotFoundError("model has not been generated yet, build model first")
        with open(engine_path2, "rb") as f:
            serialized_engine2 = f.read()
        engine2 = runtime.deserialize_cuda_engine(serialized_engine2)
        context2 = engine2.create_execution_context()
        self.context_fp16 = [context1, context2]

    def unload_tensorrt_model_fp16_ensemble(self):
        self.context_fp16 = None
        torch.cuda.empty_cache()

    def run_pytorch_model(self):
        original_output = self.sam_image_encoder.set_image(self.input_image)

    def run_tensorrt_model_fp32(self):
        tensorrt_output = \
        utils.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs,
                              stream=self.stream)[0]

    def run_tensorrt_model_fp16(self):
        tensorrt_output = \
        utils.do_inference_v2(self.context_fp16, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs,
                              stream=self.stream)[0]

    def run_tensorrt_model_ensemble(self):
        tensorrt_output = \
        utils.do_inference_v2_ensemble(*self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs,
                                       stream=self.stream)[0]

    def run_tensorrt_model_fp16_ensemble(self):
        tensorrt_output = utils.do_inference_v2_ensemble(*self.context_fp16, bindings=self.bindings, inputs=self.inputs,
                                                         outputs=self.outputs, stream=self.stream)[0]

    def time_model(self, model, warmup_iters, measure_iters):
        # Warmup loop
        for _ in range(warmup_iters):
            model()

        # Measurement loop
        start_time = timeit.default_timer()
        for _ in range(measure_iters):
            model()
        end_time = timeit.default_timer()
        elapsed_time = end_time - start_time

        avg_fps = measure_iters / elapsed_time
        avg_time = elapsed_time / measure_iters

        return avg_fps, avg_time

    def test_models(self, warmup_iters=5, measure_iters=50):
        self.__test_models(warmup_iters, measure_iters)
        torch.cuda.empty_cache()

    def __test_models(self, warmup_iters=5, measure_iters=50):
        # Timing the models
        self.load_pytorch_model()
        fps1, times1 = self.time_model(self.run_pytorch_model, warmup_iters, measure_iters)
        self.unload_pytorch_model()

        if self.model_type != "vit_h":
            self.load_tensorrt_model_fp32()
            self.inputs, self.outputs, self.bindings, self.stream = utils.allocate_buffers(self.context.engine, 1)
            utils.load_img_to_input_buffer(self.input_image_processed, pagelocked_buffer=self.inputs[0].host)
            fps2, times2 = self.time_model(self.run_tensorrt_model_fp32, warmup_iters, measure_iters)
            self.unload_tensorrt_model_fp32()

            self.load_tensorrt_model_fp16()
            self.inputs, self.outputs, self.bindings, self.stream = utils.allocate_buffers(self.context_fp16.engine, 1)
            utils.load_img_to_input_buffer(self.input_image_processed.half(), pagelocked_buffer=self.inputs[0].host)
            fps3, times3 = self.time_model(self.run_tensorrt_model_fp16, warmup_iters, measure_iters)
            self.unload_tensorrt_model_fp16()
        else:
            self.load_tensorrt_model_ensemble()
            inputs1, outputs1, inputs2, outputs2, bindings1, bindings2, stream = utils.allocate_buffers_ensemble(
                self.context[0].engine, self.context[0].engine, 1)
            utils.load_img_to_input_buffer(self.input_image_processed, pagelocked_buffer=inputs1[0].host)
            self.inputs, self.outputs, self.bindings, self.stream = [inputs1, inputs2], [outputs1, outputs2], [
                bindings1, bindings2], stream
            fps2, times2 = self.time_model(self.run_tensorrt_model_ensemble, warmup_iters, measure_iters)
            self.unload_tensorrt_model_ensemble()

            self.load_tensorrt_model_fp16_ensemble()
            inputs1, outputs1, inputs2, outputs2, bindings1, bindings2, stream = utils.allocate_buffers_ensemble(
                self.context_fp16[0].engine, self.context_fp16[0].engine, 1)
            utils.load_img_to_input_buffer(self.input_image_processed.half(), pagelocked_buffer=inputs1[0].host)
            self.inputs, self.outputs, self.bindings, self.stream = [inputs1, inputs2], [outputs1, outputs2], [
                bindings1, bindings2], stream
            fps3, times3 = self.time_model(self.run_tensorrt_model_fp16_ensemble, warmup_iters, measure_iters)
            self.unload_tensorrt_model_fp16_ensemble()

        self.add_to_table("PyTorch model", fps1, times1, fps1, times1)
        self.add_to_table("TensorRT model", fps2, times2, fps1, times1)
        self.add_to_table("TensorRT FP16 model", fps3, times3, fps1, times1)
        self.print_stats()
