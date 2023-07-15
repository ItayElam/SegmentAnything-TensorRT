from segment_anything import sam_model_registry, SamPredictor
import src.utils as utils
import tensorrt as trt
import torch
import os
import onnxruntime as ort


class InferenceEngine:
    def __init__(self, pth_path, trt_model_1: str, trt_model_2=None):
        if not os.path.isfile(pth_path):
            raise FileNotFoundError(f"couldnt find file\n{pth_path}")
        if os.path.getsize(pth_path) == 2564550879:
            if trt_model_1.endswith(".onnx"):
                self.currentModel = InferOnnx(pth_path, trt_model_1, trt_model_2, "vit_h")
            else:
                self.currentModel = InferSamH(pth_path, trt_model_1, trt_model_2, "vit_h")
        elif os.path.getsize(pth_path) == 1249524607:
            if trt_model_1.endswith(".onnx"):
                self.currentModel = InferOnnx(pth_path, trt_model_1, None, "vit_l")
            else:
                self.currentModel = InferSamBL(pth_path, trt_model_1, "vit_l")
        elif os.path.getsize(pth_path) == 375042383:
            if trt_model_1.endswith(".onnx"):
                self.currentModel = InferOnnx(pth_path, trt_model_1, None, "vit_b")
            else:
                self.currentModel = InferSamBL(pth_path, trt_model_1, "vit_b")
        else:
            raise Exception("File size does not match any known sam model sizes ")

    def __call__(self, input_image, input_point, input_label):
        masks = self.currentModel.infer(input_image, input_point, input_label)
        output_image = utils.postprocess_masks(input_image, masks)
        return output_image


class InferOnnx:
    def __init__(self, pth_path, onnx_model_1, onnx_model_2, model_type):
        self.onnx_model_1 = onnx_model_1
        self.onnx_model_2 = onnx_model_2
        sam = sam_model_registry[model_type](checkpoint=pth_path)
        self.predictor = SamPredictor(sam.to('cuda'))
        del sam
        self.session_1 = ort.InferenceSession(self.onnx_model_1, providers=['CUDAExecutionProvider'])
        if self.onnx_model_2 is not None:
            self.session_2 = ort.InferenceSession(self.onnx_model_2, providers=['CUDAExecutionProvider'])

    def infer(self, input_image, input_point, input_label):
        pixel_mean = torch.tensor([123.675, 116.28, 103.53])
        pixel_std = torch.tensor([58.395, 57.12, 57.375])
        img_size = 1024
        input_for_onnx = utils.preprocess_image(input_image, 1024, "cpu", pixel_mean, pixel_std, img_size).numpy()
        if self.onnx_model_2 is not None:

            output = self.session_2.run(None, {
                "intermediate_input": self.session_1.run(None, {"input_1": input_for_onnx}, )[0]}, )[0]
        else:
            output = self.session_1.run(None, {"input_1": input_for_onnx}, )[0]

        output = output.reshape((1, 256, 64, 64))

        self.predictor.set_image(input_image, embeddings=torch.tensor(output).to("cuda"))
        masks, scores, logits = self.predictor.predict(point_coords=input_point, point_labels=input_label,
                                                       multimask_output=False)
        return masks


class InferSamH:
    def __init__(self, pth_path, trt_model_1, trt_model_2, model_type):
        sam = sam_model_registry[model_type](checkpoint=pth_path)
        self.predictor = SamPredictor(sam.to('cuda'))
        del sam
        TRT_LOGGER = trt.Logger()
        runtime = trt.Runtime(TRT_LOGGER)

        if not os.path.isfile(trt_model_1):
            raise FileNotFoundError(f"Could not find model in path\n{trt_model_1}")
        with open(trt_model_1, "rb") as f:
            serialized_engine1 = f.read()
        engine1 = runtime.deserialize_cuda_engine(serialized_engine1)
        self.context1 = engine1.create_execution_context()

        if not os.path.isfile(trt_model_2):
            raise FileNotFoundError(f"Could not find model in path\n{trt_model_2}")
        with open(trt_model_2, "rb") as f:
            serialized_engine2 = f.read()
        engine2 = runtime.deserialize_cuda_engine(serialized_engine2)
        self.context2 = engine2.create_execution_context()

        self.inputs1, self.outputs1, self.inputs2, self.outputs2, self.bindings1, self.bindings2, self.stream = utils.allocate_buffers_ensemble(
            engine1, engine2, 1)

    def infer(self, input_image, input_point, input_label):
        pixel_mean = torch.tensor([123.675, 116.28, 103.53])
        pixel_std = torch.tensor([58.395, 57.12, 57.375])
        img_size = 1024
        input_for_onnx = utils.preprocess_image(input_image, 1024, "cpu", pixel_mean, pixel_std, img_size)

        utils.load_img_to_input_buffer(input_for_onnx, pagelocked_buffer=self.inputs1[0].host)
        [output] = utils.do_inference_v2_ensemble(self.context1, self.context2,
                                                  bindings=[self.bindings1, self.bindings2],
                                                  inputs=[self.inputs1, self.inputs2],
                                                  outputs=[self.outputs1, self.outputs2],
                                                  stream=self.stream)
        output = output.reshape((1, 256, 64, 64))

        self.predictor.set_image(input_image, embeddings=torch.tensor(output).to("cuda"))
        masks, scores, logits = self.predictor.predict(point_coords=input_point, point_labels=input_label,
                                                       multimask_output=False)

        return masks


class InferSamBL:
    def __init__(self, pth_path, trt_model, model_type):
        sam = sam_model_registry[model_type](checkpoint=pth_path)
        self.predictor = SamPredictor(sam.to('cuda'))
        del sam
        TRT_LOGGER = trt.Logger()
        runtime = trt.Runtime(TRT_LOGGER)

        if not os.path.isfile(trt_model):
            raise FileNotFoundError(f"Could not find model in path\n{trt_model}")
        with open(trt_model, "rb") as f:
            serialized_engine = f.read()

        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = engine.create_execution_context()

        self.inputs, self.outputs, self.bindings, self.stream = utils.allocate_buffers(engine, max_batch_size=1)

    def infer(self, input_image, input_point, input_label):
        pixel_mean = torch.tensor([123.675, 116.28, 103.53])
        pixel_std = torch.tensor([58.395, 57.12, 57.375])
        img_size = 1024
        input_for_onnx = utils.preprocess_image(input_image, 1024, "cpu", pixel_mean, pixel_std, img_size)

        utils.load_img_to_input_buffer(input_for_onnx, pagelocked_buffer=self.inputs[0].host)
        [output] = utils.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs,
                                         stream=self.stream)
        output = output.reshape((1, 256, 64, 64))

        self.predictor.set_image(input_image, embeddings=torch.tensor(output).to("cuda"))
        masks, scores, logits = self.predictor.predict(point_coords=input_point, point_labels=input_label,
                                                       multimask_output=False)

        return masks
