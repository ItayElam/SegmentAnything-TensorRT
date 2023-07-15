import os.path

import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import tensorrt as trt
from torchvision.transforms.functional import resize, to_pil_image
import torch
from torch.nn import functional as F
import cv2


class ImageClickHandler:
    def __init__(self, image):
        self.image = image
        self.original_image = image.copy()
        self.clicked_coordinates = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.image = self.original_image.copy()
            self.clicked_coordinates = (x, y)
            cv2.circle(self.image, self.clicked_coordinates, 3, (0, 0, 255), -1)

    def run(self):
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_callback)

        while True:
            cv2.imshow("Image", self.image)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:
                break

        cv2.destroyAllWindows()
        return self.clicked_coordinates


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, size=0):
        self.host = host_mem
        self.device = device_mem
        self.size = size

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def choose_point(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"could not find {image_path}")
    image = cv2.imread(image_path)
    click_handler = ImageClickHandler(image)
    clicked_coords = click_handler.run()
    return np.array([clicked_coords])


def postprocess_masks(image, mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30, 144, 255])
    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w, 1)
    mask_image = np.where(mask, image * 0.4 + mask * color.reshape(1, 1, -1) * 0.6, image)
    return mask_image


def preprocess_image(image: np.ndarray, target_length: int, device, pixel_mean, pixel_std, img_size):
    scale = target_length * 1.0 / max(image.shape[0], image.shape[1])
    newh, neww = image.shape[0] * scale, image.shape[1] * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    target_size = newh, neww

    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)

    input_image = np.array(resize(to_pil_image(image), target_size))
    input_image_torch = torch.as_tensor(input_image).to(device)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    # Normalize colors
    input_image_torch = (input_image_torch - pixel_mean.to(device)) / pixel_std.to(device)

    # Pad
    h, w = input_image_torch.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    input_image_torch = F.pad(input_image_torch, (0, padw, 0, padh))
    return input_image_torch


def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def do_inference_v2_ensemble(context1, context2, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs[0]]
    # Run inference.
    context1.execute_async_v2(bindings=bindings[0], stream_handle=stream.handle)

    stream.synchronize()
    [cuda.memcpy_dtod(inp.device, out.device, inp.size) for out, inp in zip(outputs[0], inputs[1])]

    stream.synchronize()
    context2.execute_async_v2(bindings=bindings[1], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs[1]]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs[1]]


def allocate_buffers(engine, max_batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        # print(binding, engine.get_binding_shape(binding))
        # print(engine.get_binding_shape(binding))
        size = trt.volume(engine.get_binding_shape(binding)) * max_batch_size
        if size < 0 and engine.binding_is_input(binding):
            size = trt.volume(engine.get_profile_shape(0, binding)[2]) * max_batch_size
        if binding == "masks":
            size = 534 * 800 * max_batch_size
        # print(binding, engine.get_binding_shape(binding), engine.get_profile_shape(0, binding)[2])
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def allocate_buffers_ensemble(engine1, engine2, max_batch_size, intermediate=None):
    inputs1 = []
    outputs1 = []
    inputs2 = []
    outputs2 = []
    bindings1 = []
    bindings2 = []
    stream = cuda.Stream()

    # Allocating buffers for engine1
    for binding in engine1:
        size = trt.volume(engine1.get_binding_shape(binding)) * max_batch_size
        if size < 0 and engine1.binding_is_input(binding):
            size = trt.volume(engine1.get_profile_shape(0, binding)[2]) * max_batch_size
        dtype = trt.nptype(engine1.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings1.append(int(device_mem))
        if engine1.binding_is_input(binding):
            inputs1.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs1.append(HostDeviceMem(host_mem, device_mem))

    # Allocating buffers for engine2
    for binding in engine2:
        size = trt.volume(engine2.get_binding_shape(binding)) * max_batch_size
        if size < 0 and engine2.binding_is_input(binding):
            size = trt.volume(engine2.get_profile_shape(0, binding)[2]) * max_batch_size
        dtype = trt.nptype(engine2.get_binding_dtype(binding))

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings2.append(int(device_mem))
        if engine2.binding_is_input(binding):
            inputs2.append(HostDeviceMem(host_mem, device_mem, size * (4 if dtype == np.float32 else 2)))
        else:
            outputs2.append(HostDeviceMem(host_mem, device_mem))

    return inputs1, outputs1, inputs2, outputs2, bindings1, bindings2, stream


def load_img_to_input_buffer(img, pagelocked_buffer):
    preprocessed = np.asarray(img).astype(np.float32).ravel()
    np.copyto(pagelocked_buffer, preprocessed)
