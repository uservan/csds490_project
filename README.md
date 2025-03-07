# Lighting up my World

by Michael Carlstrom (rmc170), Wang Yang (wxy320), Chenxing Liu (cxl1697)

## Project Overview:

The project goal is to create and evaulate various methods of enhancing images in low-light conditions.

Typically by default when an image is taken it stores the raw rgb values in the image. To the human eye our perception struggles to determine sublte differenes so objects that are geuniley a different color can easily get lost in a dark image. Using image processing and machine learning tools can help overcome human eye limitations and make images more presentable to humans.

This project would be used by various camera and smartphone manafucatures to add in the ability to take better low-light condition pictures thus making a better product and sales proposition.

The benefit will be the greater availability of cameras which take good quality images in low lighting conditions. 


## Problem Statement:

How to fix images taken in dark lighting conditions?

Examples from "Beyond Brightening Low-light Images"

![alt text](images/11263_2020_1407_Fig19_HTML.png)

The traditional approach is using histogram equalization. This typically is quite limited and works better on balancing out brightness rather than raising the brightness of an extremely dark image.

A more modern approach is to use a Generative Adversial Network (GAN) and train a network to brighten images. This approach can work quite well on training data but, can struggle to generalize to other images depending on the tuning and dataset chosen.

## Objectives:

Brighten up images taken in various degrees of darkness.

## 5 Methodology:

In this section, a brief outline of the methodologies we will use in our project will be provided. Specifically, we will explore basic and state-of-the-art approaches categorized into three main types: **traditional methods**, **dehazing-based methods**, and **GAN-based methods**. The traditional methods are further divided into **histogram-based methods** and **Retinex theory-based methods**. Below, we provide a brief introduction to each category.

### Responsibility Assignement
* **Chenxing Liu**: traditional methods;
* **Wang Yang**: dehazing-based methods;
* **Michael Carlstrom**: GAN-based methods.

### 5.1 Traditional Method
#### 5.1.1 Histogram-based Method

The histogram-based methods offer a straightforward and intuitive way to enhance image by directly adjusting its intensity distribution. This approach is particularly effective for image enhancement in low-light conditions, where pixel intensities are predominantly concentrated in the lower range. 

* **Histogram Equalization (HE)**
  
  HE enhances an image by performing global contrast stretching. It computes the histogram of the entire image and redistributes these pixel intensities to achieve a more uniform histogram.
   
* **Adaptive Histogram Equalization (AHE)**
  
  HE performs global contrast stretching, which may cause issues such as posterization (or banding) and noise amplification. To mitigate these issues, Adaptive Histogram Equalization (AHE) divides the image into multiple non-overlapping sub-regions (tiles) and applies histogram equalization independently to each region.
  
* **Contrast Limited Adaptive Histogram Equalization (CLAHE)**
  
  CLAHE further extends AHE by introducing a contrast limiting threshold (Clip Limit) to prevent excessive enhancement of certain pixel intensities and applying bilinear interpolation to ensure smooth transitions. These effectively mitigate the issue of noise amplification and reduce blocking artifacts.
  
#### 5.1.2 Retinex-based Method

  Retinex theory suggests that a perceived image $f(x,y)$ is formed by the interaction of illumination and reflectance:
  
  $$f(x,y)=i(x,y) \cdot r(x,y),$$
  
  where $i(x,y)$ and $r(x,y)$ represents the **illumination component** and the **reflectance component** respectively. In the context of low-light image enhancement, the fundamental idea behind Retinex-based methods is that low-light images suffer from bad visibility due to insufficient illumination condition $i(x,y)$. Therefore, by separating the illumination $i(x,y)$ and reflectance $r(x,y)$, the influence of illumination can be eliminated, allowing us to further restore the true appearance of objects $r(x,y)$. Different algorithms adopt different approach to estimate $i(x,y)$.

* **Single-Scale Retinex (SSR)**
  
  The illumination component $i(x,y)$ primarily contains low-frequency information, as it represents large-scale brightness variations. In contrast, the reflectance component $r(x,y)$ mainly consists of high-frequency details, including textures and fine structures. Single-scale retinex estimates the illumination $i(x,y)$ by convolve the original image with a low-pass Gaussian filter. By eliminating the estimated illumination component, the reflectance can be obtained:

  $$\ln{r(x,y)}=\ln{f(x,y)}-\ln{i(x,y)}=\ln{f(x,y)}-\ln{f(x,y)\ast G(x,y,\sigma)},$$

  where $G(x,y,\sigma)$ represents a low-pass Gaussian kernel with standard deviation $\sigma$. Finally, contrast stretching is applied to $\ln{⁡r(x,y)}$ to obtain the enhanced image.
  
* **Multi-Scale Retinex (MSR)**
  
  Multi-scale Retinex (MSR) extends SSR by applying multiple low-pass Gaussian filters with different standard deviation, and combining the results through a weighted average:
  
  $$\ln{r(x,y)}=\sum_{k=1}^{n}w_k[\ln{f(x,y)}-\ln{i(x,y)}=\ln{f(x,y)}-\ln{f(x,y)\ast G(x,y,\sigma_k)}],$$

  where $G(x,y,\sigma_k)$ represents low-pass Gaussian kernel with different standard deviation $\sigma_k$.

* **LIME**
  
  LIME enhanced low-light images by directly estimating the illumination map. It approximates the illumination component by taking the maximum value across the RGB channels:
  
  $$r(x,y)=max_{c\in\lbrace R,G,B \rbrace}f_c(x,y).$$

  To ensure spatial smoothness, additional constraints are applied to refine the illumination map. Finally, the estimated illumination is removed from the original image, yielding the enhanced image with improved visibility and contrast.

  
### 5.2 Dehazing-based Method

Low-light images can also be enhanced using dehazing-based methods. The key idea behind this approach is that the inverse image of a low-light image shares similar characteristics with a hazy image. Inspired by this, the method proposed in ***“Fast Efficient Algorithm for Enhancement of Low Light Video”*** formulate low-light enhancement as an image dehazing problem. The method consists of the following steps:

1. Invert the original image, transforming it into a form that resembles a hazy image.

2. Apply a dehazing algorithm to enhance visibility by removing haze-like effects.

3. Invert the dehazed image again to obtain the final enhanced low-light image

### 5.2 GAN-based Method

  Generative Adversarial Networks (GANs) are a type of deep learning framework for generation tasks and have proven successful in image enhancement. Given dataset composed of low light images and normal light images, a GAN can be trained to automatically generate the normal-light version of the low-light images, outperforming traditional techniques in terms of visually appealing and image quality.
  
**EnlightenGAN**, proposed in ***"EnlightenGAN: Deep Light Enhancement without Paired Supervision"***, introduces a self-supervised approach that does not require paired low/normal-light images for training. The method introduces several key innovations, including:

* **A global-local discriminator structure** for improved feature representation.
* **A self-regularized perceptual loss function** to guide natural-looking enhancement.
* **An attention mechanism** to better capture important details.
EnlightenGAN is demonstrated to be easily adaptable to enhancing real-world images from various domains. And we will implement and evaluate this method in our project.

## Tools and Technologies:

## Image for the Project:

## Timeline

## Work Division
