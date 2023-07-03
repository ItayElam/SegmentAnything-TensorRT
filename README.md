# Segment Anything TensorRT 
introduction
------------
Welcome to the TensorRT implementation of the "Segment Anything" model!

Overview:
------------
This repository contains the implementation of the "Segment Anything" model in TensorRT. While I found existing implementations for vit_b and vit_l, I couldn't find one for vit_h. Therefore, to the best of my knowledge, this is the first implementation available online that covers all three model types.

Features:
-----------
- Performance Evaluation: Explore the performance of the implemented model.
- Visual Comparisons: Find visual comparisons of the model's output.

Upcoming Updates:
----------------------
I'm actively working on improving the code to make it more user-friendly. Once the revisions are complete, I will release the updated code, which will include the following:
- Code for converting the models in a user-friendly format.
- Detailed accuracy and performance analysis for evaluating the model.

Stay tuned for the upcoming updates, and feel free to contribute and provide feedback!

## Performance

### Benchmarking on RTX 3090

#### Performance Comparison for vit_b
| Model              | Average FPS | Average Time (sec) | Relative FPS | Relative Time (%) |
|------------------- |-------------|--------------------|--------------|-------------------|
| PyTorch model      | 9.96        | 0.100417           | 1.0          | 100.0             |
| TensorRT model     | 15.24       | 0.065603           | 1.53         | 65.33             |
| TensorRT FP16 model| 29.32       | 0.034104           | 2.94         | 33.96             |

#### Performance Comparison for vit_l
| Model              | Average FPS | Average Time (sec) | Relative FPS | Relative Time (%) |
|------------------- |-------------|--------------------|--------------|-------------------|
| PyTorch model      | 3.91        | 0.255552           | 1.0          | 100.0             |
| TensorRT model     | 4.81        | 0.208019           | 1.23         | 81.4              |
| TensorRT FP16 model| 11.09       | 0.090139           | 2.84         | 35.27             |

#### Performance Comparison for vit_h
| Model              | Average FPS | Average Time (sec) | Relative FPS | Relative Time (%) |
|------------------- |-------------|--------------------|--------------|-------------------|
| PyTorch model      | 2.22        | 0.45045            | 1.0          | 100.0             |
| TensorRT model     | 2.37        | 0.421377           | 1.07         | 93.55             |
| TensorRT FP16 model| 5.97        | 0.167488           | 2.69         | 37.18             |

## Visualizations
### original image
<p float="left">
  <img src="images/original_image.jpg" alt="Original image" width="30%" />
</p>

### vit_b

<p float="left">
  <img src="images/vit_b_Mask_Original.png" alt="Original vit_b" width="30%" />
  <img src="images/vit_b_Mask_TensorRT.png" alt="TensorRT FP32 vit_b" width="30%" /> 
  <img src="images/vit_b_Mask_TensorRT_FP16.png" alt="TensorRT FP16 vit_b" width="30%" />
</p>

### vit_l

<p float="left">
  <img src="images/vit_l_Mask_Original.png" alt="Original vit_l" width="30%" />
  <img src="images/vit_l_Mask_TensorRT.png" alt="TensorRT FP32 vit_l" width="30%" /> 
  <img src="images/vit_l_Mask_TensorRT_FP16.png" alt="TensorRT FP16 vit_l" width="30%" />
</p>

### vit_h

<p float="left">
  <img src="images/vit_h_Mask_Original.png" alt="Original vit_h" width="30%" />
  <img src="images/vit_h_Mask_TensorRT.png" alt="TensorRT FP32 vit_h" width="30%" /> 
  <img src="images/vit_h_Mask_TensorRT_FP16.png" alt="TensorRT FP16 vit_h" width="30%" />
</p>