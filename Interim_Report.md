# Finished

## 1 Traditional Methods

In our proposal, we plan to implement traditional methods for image enhancement in low-light conditions as a baseline for comparison with other advanced approaches. Specifically, we aim to implement methods based on histogram equalization and Retinex theory. So far, we have completed the **histogram equalization** part. Experiments have been conducted on the dataset, and ablation studies have been carried out to investigate the impact of different factors on image enhancement performance.

### Results and Visualization

- **Histogram Equalization (HE)**


- **Adaptive Histogram Equalization (AHE)**


- **Contrast Limited Adaptive Histogram Equalization (CLAHE)**



### Ablation Study

## 2 Dehazing-based Low-Light Image Enhancement

Inspired by the observation that the inverted form of a low-light image resembles a hazy image, we implemented a dehazing-based method to enhance low-light images. This approach is motivated by the paper *"Fast Efficient Algorithm for Enhancement of Low Light Video"* and leverages the classic Dark Channel Prior (DCP) dehazing technique.

### Results and Visualization

We compare three versions of the image:

- **Left**: Original low-light input  
- **Middle**: Ground truth or reference high-brightness image  
- **Right**: Our dehazing-enhanced result


<div style="display: flex; justify-content: space-between; gap: 10px;">
  <div style="text-align: center;">
    <img src="images/dehaze/low22.png" alt="Low-light" width="250"/>
    <p style="font-size: 14px;">Low-light Input</p>
  </div>
  <div style="text-align: center;">
    <img src="images/dehaze/high22.png" alt="Ground Truth" width="250"/>
    <p style="font-size: 14px;">Original Bright Image</p>
  </div>
  <div style="text-align: center;">
    <img src="images/dehaze/enhanced.jpg" alt="Enhanced" width="250"/>
    <p style="font-size: 14px;">Dehazing-based Enhanced</p>
  </div>
</div>

As shown above, our method significantly improves the brightness and visibility in dark regions. While minor color shifts may occur due to atmospheric light estimation, the result offers a clear enhancement in terms of detail and contrast.


# Next Stage
## Evaluation Methods
- One Next Stage is to finish the evaluation pipiline and evaluate all the methods
