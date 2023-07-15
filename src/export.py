import os

from segment_anything import sam_model_registry
import torch
import onnxruntime

import sam_modification.ImageEncoderModels as ImageEncoderModels

onnxruntime.set_default_logger_severity(3)


class ExportSAM:
    def __init__(self, pth_path, device="cpu"):
        if not os.path.isfile(pth_path):
            raise FileNotFoundError(f"couldnt find file\n{pth_path}")
        if "vit_h" in os.path.split(pth_path)[1]:
            self.currentModel = ExportSamH(pth_path, device)
        elif "vit_l" in os.path.split(pth_path)[1]:
            self.currentModel = ExportSamBL(pth_path, "vit_l", device)
        elif "vit_b" in os.path.split(pth_path)[1]:
            self.currentModel = ExportSamBL(pth_path, "vit_b", device)
        else:
            raise Exception("File size does not match any known sam model sizes ")

    def __call__(self, model_precision):
        if model_precision != "fp32" and model_precision != "fp16" and model_precision != "both":
            raise Exception("model_precision is not legal, choose from 'fp32', 'fp16', 'both'")
        os.makedirs(os.path.join("exported_models", self.currentModel.model_type), exist_ok=True)
        self.currentModel.export_onnx()
        self.currentModel.export_trt(model_precision)


class ExportSamH:
    def __init__(self, model_path, device):
        self.sam_checkpoint = model_path
        self.device = device
        self.model_type = "vit_h"

    def export_onnx(self):
        first_half = ["image_encoder.pos_embed", "image_encoder.patch_embed.proj.weight",
                      "image_encoder.patch_embed.proj.bias", "image_encoder.blocks.0.norm1.weight",
                      "image_encoder.blocks.0.norm1.bias", "image_encoder.blocks.0.attn.rel_pos_h",
                      "image_encoder.blocks.0.attn.rel_pos_w", "image_encoder.blocks.0.attn.qkv.weight",
                      "image_encoder.blocks.0.attn.qkv.bias", "image_encoder.blocks.0.attn.proj.weight",
                      "image_encoder.blocks.0.attn.proj.bias", "image_encoder.blocks.0.norm2.weight",
                      "image_encoder.blocks.0.norm2.bias", "image_encoder.blocks.0.mlp.lin1.weight",
                      "image_encoder.blocks.0.mlp.lin1.bias", "image_encoder.blocks.0.mlp.lin2.weight",
                      "image_encoder.blocks.0.mlp.lin2.bias", "image_encoder.blocks.1.norm1.weight",
                      "image_encoder.blocks.1.norm1.bias", "image_encoder.blocks.1.attn.rel_pos_h",
                      "image_encoder.blocks.1.attn.rel_pos_w", "image_encoder.blocks.1.attn.qkv.weight",
                      "image_encoder.blocks.1.attn.qkv.bias", "image_encoder.blocks.1.attn.proj.weight",
                      "image_encoder.blocks.1.attn.proj.bias", "image_encoder.blocks.1.norm2.weight",
                      "image_encoder.blocks.1.norm2.bias", "image_encoder.blocks.1.mlp.lin1.weight",
                      "image_encoder.blocks.1.mlp.lin1.bias", "image_encoder.blocks.1.mlp.lin2.weight",
                      "image_encoder.blocks.1.mlp.lin2.bias", "image_encoder.blocks.2.norm1.weight",
                      "image_encoder.blocks.2.norm1.bias", "image_encoder.blocks.2.attn.rel_pos_h",
                      "image_encoder.blocks.2.attn.rel_pos_w", "image_encoder.blocks.2.attn.qkv.weight",
                      "image_encoder.blocks.2.attn.qkv.bias", "image_encoder.blocks.2.attn.proj.weight",
                      "image_encoder.blocks.2.attn.proj.bias", "image_encoder.blocks.2.norm2.weight",
                      "image_encoder.blocks.2.norm2.bias", "image_encoder.blocks.2.mlp.lin1.weight",
                      "image_encoder.blocks.2.mlp.lin1.bias", "image_encoder.blocks.2.mlp.lin2.weight",
                      "image_encoder.blocks.2.mlp.lin2.bias", "image_encoder.blocks.3.norm1.weight",
                      "image_encoder.blocks.3.norm1.bias", "image_encoder.blocks.3.attn.rel_pos_h",
                      "image_encoder.blocks.3.attn.rel_pos_w", "image_encoder.blocks.3.attn.qkv.weight",
                      "image_encoder.blocks.3.attn.qkv.bias", "image_encoder.blocks.3.attn.proj.weight",
                      "image_encoder.blocks.3.attn.proj.bias", "image_encoder.blocks.3.norm2.weight",
                      "image_encoder.blocks.3.norm2.bias", "image_encoder.blocks.3.mlp.lin1.weight",
                      "image_encoder.blocks.3.mlp.lin1.bias", "image_encoder.blocks.3.mlp.lin2.weight",
                      "image_encoder.blocks.3.mlp.lin2.bias", "image_encoder.blocks.4.norm1.weight",
                      "image_encoder.blocks.4.norm1.bias", "image_encoder.blocks.4.attn.rel_pos_h",
                      "image_encoder.blocks.4.attn.rel_pos_w", "image_encoder.blocks.4.attn.qkv.weight",
                      "image_encoder.blocks.4.attn.qkv.bias", "image_encoder.blocks.4.attn.proj.weight",
                      "image_encoder.blocks.4.attn.proj.bias", "image_encoder.blocks.4.norm2.weight",
                      "image_encoder.blocks.4.norm2.bias", "image_encoder.blocks.4.mlp.lin1.weight",
                      "image_encoder.blocks.4.mlp.lin1.bias", "image_encoder.blocks.4.mlp.lin2.weight",
                      "image_encoder.blocks.4.mlp.lin2.bias", "image_encoder.blocks.5.norm1.weight",
                      "image_encoder.blocks.5.norm1.bias", "image_encoder.blocks.5.attn.rel_pos_h",
                      "image_encoder.blocks.5.attn.rel_pos_w", "image_encoder.blocks.5.attn.qkv.weight",
                      "image_encoder.blocks.5.attn.qkv.bias", "image_encoder.blocks.5.attn.proj.weight",
                      "image_encoder.blocks.5.attn.proj.bias", "image_encoder.blocks.5.norm2.weight",
                      "image_encoder.blocks.5.norm2.bias", "image_encoder.blocks.5.mlp.lin1.weight",
                      "image_encoder.blocks.5.mlp.lin1.bias", "image_encoder.blocks.5.mlp.lin2.weight",
                      "image_encoder.blocks.5.mlp.lin2.bias", "image_encoder.blocks.6.norm1.weight",
                      "image_encoder.blocks.6.norm1.bias", "image_encoder.blocks.6.attn.rel_pos_h",
                      "image_encoder.blocks.6.attn.rel_pos_w", "image_encoder.blocks.6.attn.qkv.weight",
                      "image_encoder.blocks.6.attn.qkv.bias", "image_encoder.blocks.6.attn.proj.weight",
                      "image_encoder.blocks.6.attn.proj.bias", "image_encoder.blocks.6.norm2.weight",
                      "image_encoder.blocks.6.norm2.bias", "image_encoder.blocks.6.mlp.lin1.weight",
                      "image_encoder.blocks.6.mlp.lin1.bias", "image_encoder.blocks.6.mlp.lin2.weight",
                      "image_encoder.blocks.6.mlp.lin2.bias", "image_encoder.blocks.7.norm1.weight",
                      "image_encoder.blocks.7.norm1.bias", "image_encoder.blocks.7.attn.rel_pos_h",
                      "image_encoder.blocks.7.attn.rel_pos_w", "image_encoder.blocks.7.attn.qkv.weight",
                      "image_encoder.blocks.7.attn.qkv.bias", "image_encoder.blocks.7.attn.proj.weight",
                      "image_encoder.blocks.7.attn.proj.bias", "image_encoder.blocks.7.norm2.weight",
                      "image_encoder.blocks.7.norm2.bias", "image_encoder.blocks.7.mlp.lin1.weight",
                      "image_encoder.blocks.7.mlp.lin1.bias", "image_encoder.blocks.7.mlp.lin2.weight",
                      "image_encoder.blocks.7.mlp.lin2.bias", "image_encoder.blocks.8.norm1.weight",
                      "image_encoder.blocks.8.norm1.bias", "image_encoder.blocks.8.attn.rel_pos_h",
                      "image_encoder.blocks.8.attn.rel_pos_w", "image_encoder.blocks.8.attn.qkv.weight",
                      "image_encoder.blocks.8.attn.qkv.bias", "image_encoder.blocks.8.attn.proj.weight",
                      "image_encoder.blocks.8.attn.proj.bias", "image_encoder.blocks.8.norm2.weight",
                      "image_encoder.blocks.8.norm2.bias", "image_encoder.blocks.8.mlp.lin1.weight",
                      "image_encoder.blocks.8.mlp.lin1.bias", "image_encoder.blocks.8.mlp.lin2.weight",
                      "image_encoder.blocks.8.mlp.lin2.bias", "image_encoder.blocks.9.norm1.weight",
                      "image_encoder.blocks.9.norm1.bias", "image_encoder.blocks.9.attn.rel_pos_h",
                      "image_encoder.blocks.9.attn.rel_pos_w", "image_encoder.blocks.9.attn.qkv.weight",
                      "image_encoder.blocks.9.attn.qkv.bias", "image_encoder.blocks.9.attn.proj.weight",
                      "image_encoder.blocks.9.attn.proj.bias", "image_encoder.blocks.9.norm2.weight",
                      "image_encoder.blocks.9.norm2.bias", "image_encoder.blocks.9.mlp.lin1.weight",
                      "image_encoder.blocks.9.mlp.lin1.bias", "image_encoder.blocks.9.mlp.lin2.weight",
                      "image_encoder.blocks.9.mlp.lin2.bias", "image_encoder.blocks.10.norm1.weight",
                      "image_encoder.blocks.10.norm1.bias", "image_encoder.blocks.10.attn.rel_pos_h",
                      "image_encoder.blocks.10.attn.rel_pos_w", "image_encoder.blocks.10.attn.qkv.weight",
                      "image_encoder.blocks.10.attn.qkv.bias", "image_encoder.blocks.10.attn.proj.weight",
                      "image_encoder.blocks.10.attn.proj.bias", "image_encoder.blocks.10.norm2.weight",
                      "image_encoder.blocks.10.norm2.bias", "image_encoder.blocks.10.mlp.lin1.weight",
                      "image_encoder.blocks.10.mlp.lin1.bias", "image_encoder.blocks.10.mlp.lin2.weight",
                      "image_encoder.blocks.10.mlp.lin2.bias", "image_encoder.blocks.11.norm1.weight",
                      "image_encoder.blocks.11.norm1.bias", "image_encoder.blocks.11.attn.rel_pos_h",
                      "image_encoder.blocks.11.attn.rel_pos_w", "image_encoder.blocks.11.attn.qkv.weight",
                      "image_encoder.blocks.11.attn.qkv.bias", "image_encoder.blocks.11.attn.proj.weight",
                      "image_encoder.blocks.11.attn.proj.bias", "image_encoder.blocks.11.norm2.weight",
                      "image_encoder.blocks.11.norm2.bias", "image_encoder.blocks.11.mlp.lin1.weight",
                      "image_encoder.blocks.11.mlp.lin1.bias", "image_encoder.blocks.11.mlp.lin2.weight",
                      "image_encoder.blocks.11.mlp.lin2.bias", "image_encoder.blocks.12.norm1.weight",
                      "image_encoder.blocks.12.norm1.bias", "image_encoder.blocks.12.attn.rel_pos_h",
                      "image_encoder.blocks.12.attn.rel_pos_w", "image_encoder.blocks.12.attn.qkv.weight",
                      "image_encoder.blocks.12.attn.qkv.bias", "image_encoder.blocks.12.attn.proj.weight",
                      "image_encoder.blocks.12.attn.proj.bias", "image_encoder.blocks.12.norm2.weight",
                      "image_encoder.blocks.12.norm2.bias", "image_encoder.blocks.12.mlp.lin1.weight",
                      "image_encoder.blocks.12.mlp.lin1.bias", "image_encoder.blocks.12.mlp.lin2.weight",
                      "image_encoder.blocks.12.mlp.lin2.bias", "image_encoder.blocks.13.norm1.weight",
                      "image_encoder.blocks.13.norm1.bias", "image_encoder.blocks.13.attn.rel_pos_h",
                      "image_encoder.blocks.13.attn.rel_pos_w", "image_encoder.blocks.13.attn.qkv.weight",
                      "image_encoder.blocks.13.attn.qkv.bias", "image_encoder.blocks.13.attn.proj.weight",
                      "image_encoder.blocks.13.attn.proj.bias", "image_encoder.blocks.13.norm2.weight",
                      "image_encoder.blocks.13.norm2.bias", "image_encoder.blocks.13.mlp.lin1.weight",
                      "image_encoder.blocks.13.mlp.lin1.bias", "image_encoder.blocks.13.mlp.lin2.weight",
                      "image_encoder.blocks.13.mlp.lin2.bias", "image_encoder.blocks.14.norm1.weight",
                      "image_encoder.blocks.14.norm1.bias", "image_encoder.blocks.14.attn.rel_pos_h",
                      "image_encoder.blocks.14.attn.rel_pos_w", "image_encoder.blocks.14.attn.qkv.weight",
                      "image_encoder.blocks.14.attn.qkv.bias", "image_encoder.blocks.14.attn.proj.weight",
                      "image_encoder.blocks.14.attn.proj.bias", "image_encoder.blocks.14.norm2.weight",
                      "image_encoder.blocks.14.norm2.bias", "image_encoder.blocks.14.mlp.lin1.weight",
                      "image_encoder.blocks.14.mlp.lin1.bias", "image_encoder.blocks.14.mlp.lin2.weight",
                      "image_encoder.blocks.14.mlp.lin2.bias", "image_encoder.blocks.15.norm1.weight",
                      "image_encoder.blocks.15.norm1.bias", "image_encoder.blocks.15.attn.rel_pos_h",
                      "image_encoder.blocks.15.attn.rel_pos_w", "image_encoder.blocks.15.attn.qkv.weight",
                      "image_encoder.blocks.15.attn.qkv.bias", "image_encoder.blocks.15.attn.proj.weight",
                      "image_encoder.blocks.15.attn.proj.bias", "image_encoder.blocks.15.norm2.weight",
                      "image_encoder.blocks.15.norm2.bias", "image_encoder.blocks.15.mlp.lin1.weight",
                      "image_encoder.blocks.15.mlp.lin1.bias", "image_encoder.blocks.15.mlp.lin2.weight",
                      "image_encoder.blocks.15.mlp.lin2.bias"]
        with open(self.sam_checkpoint, "rb") as f:
            state_dict = torch.load(f)
        first_hald_state_dict = {}
        second_hald_state_dict = {}
        [first_hald_state_dict.update({key.replace("image_encoder.", ""): value}) for key, value in state_dict.items()
         if key in first_half]
        i = 16
        for key, value in (state_dict.items()):
            if key not in first_half and "image_encoder" in key:
                if "block" in key:
                    number = int(key.split('.')[2])
                    key = key.replace(str(number), str(number - 16))
                second_hald_state_dict.update({key.replace("image_encoder.", "").replace(str(16 + i), str(i)): value})

        first_model = ImageEncoderModels.build_first_half()
        first_model.load_state_dict(first_hald_state_dict, strict=True)
        second_model = ImageEncoderModels.build_second_half()
        second_model.load_state_dict(second_hald_state_dict, strict=True)

        dummy_input_first = torch.randn(1, 3, 1024, 1024, device=self.device)
        dummy_input_second = torch.randn(1, 64, 64, 1280, device=self.device)

        input_names_first = ["input_1"]
        output_names_first = ["intermediate_output"]
        input_names_second = ["intermediate_input"]
        output_names_second = ["output_1"]

        first_model.eval()
        second_model.eval()

        with torch.no_grad():
            torch.onnx.export(first_model, dummy_input_first,
                              f"exported_models/vit_h/sam_vit_h_embedding_first.onnx", verbose=True,
                              input_names=input_names_first, output_names=output_names_first, opset_version=17)
            torch.onnx.export(second_model, dummy_input_second,
                              f"exported_models/vit_h/sam_vit_h_embedding_second.onnx", verbose=True,
                              input_names=input_names_second, output_names=output_names_second, opset_version=17)

    def export_trt(self, model_precision):
        if model_precision == "fp32" or model_precision == "both":
            bashCommand = f"trtexec --onnx=exported_models/vit_h/sam_vit_h_embedding_first.onnx --saveEngine=exported_models/vit_h/model_fp32_1.engine"
            os.system(bashCommand)
            bashCommand = f"trtexec --onnx=exported_models/vit_h/sam_vit_h_embedding_second.onnx --saveEngine=exported_models/vit_h/model_fp32_2.engine"
            os.system(bashCommand)
        if model_precision == "fp16" or model_precision == "both":
            bashCommand = f"trtexec --onnx=exported_models/vit_h/sam_vit_h_embedding_first.onnx --fp16 --saveEngine=exported_models/vit_h/model_fp16_1.engine"
            os.system(bashCommand)
            bashCommand = f"trtexec --onnx=exported_models/vit_h/sam_vit_h_embedding_second.onnx --fp16 --saveEngine=exported_models/vit_h/model_fp16_2.engine --precisionConstraints=obey --layerPrecisions=/neck/neck.1/Sub:fp32,/neck/neck.1/Pow:fp32,/neck/neck.1/ReduceMean_1:fp32,/neck/neck.1/Add:fp32,/neck/neck.1/Sqrt:fp32,/neck/neck.1/Div:fp32,/neck/neck.1/Mul:fp32,/neck/neck.1/Add_1:fp32,/neck/neck.3/Sub:fp32,/neck/neck.3/Pow:fp32,/neck/neck.3/ReduceMean_1:fp32,/neck/neck.3/Add:fp32,/neck/neck.3/Sqrt:fp32,/neck/neck.3/Div:fp32,/neck/neck.3/Mul:fp32,/neck/neck.3/Add_1:fp32"
            os.system(bashCommand)


class ExportSamBL:
    def __init__(self, pth_path, model_type, device):
        self.sam_checkpoint = pth_path
        self.model_type = model_type
        self.device = device

    def export_onnx(self):
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        model = sam.image_encoder
        model.eval()
        dummy_input = torch.randn(1, 3, 1024, 1024, device=self.device)

        input_names = ["input_1"]
        output_names = ["output_1"]
        os.makedirs(f"exported_models/{self.model_type}", exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(model, dummy_input,
                              f"exported_models/{self.model_type}/sam_{self.model_type}_embedding.onnx", verbose=True,
                              input_names=input_names, output_names=output_names, opset_version=17)

    def export_trt(self, model_precision):

        if model_precision == "fp32" or model_precision == "both":
            bashCommand = f"trtexec --onnx=exported_models/{self.model_type}/sam_{self.model_type}_embedding.onnx --saveEngine=exported_models/{self.model_type}/model_fp32.engine"
            os.system(bashCommand)

        if model_precision == "fp16" or model_precision == "both":
            bashCommand = f"trtexec --onnx=exported_models/{self.model_type}/sam_{self.model_type}_embedding.onnx --fp16 --saveEngine=exported_models/{self.model_type}/model_fp16.engine --precisionConstraints=obey --layerPrecisions=/neck/neck.1/Sub:fp32,/neck/neck.1/Pow:fp32,/neck/neck.1/ReduceMean_1:fp32,/neck/neck.1/Add:fp32,/neck/neck.1/Sqrt:fp32,/neck/neck.1/Div:fp32,/neck/neck.1/Mul:fp32,/neck/neck.1/Add_1:fp32,/neck/neck.3/Sub:fp32,/neck/neck.3/Pow:fp32,/neck/neck.3/ReduceMean_1:fp32,/neck/neck.3/Add:fp32,/neck/neck.3/Sqrt:fp32,/neck/neck.3/Div:fp32,/neck/neck.3/Mul:fp32,/neck/neck.3/Add_1:fp32"
            os.system(bashCommand)
