# pyFusionSR
multi-spectral image fusion 

Created by Ferhat Can ATAMAN

## Tests
--- 

- [ ] Accuracy benchmarks -- save result images for 21 images. Do bench in MATLAB
  - [bench-paper](https://arxiv.org/pdf/2002.03322.pdf) 
  - [metrics](https://github.com/zhengliu6699/imageFusionMetrics)
  - [input-result](https://github.com/xingchenzhang/VIFB) 
  - [metrics2](https://github.com/noonelikechu/image-fusion-evalution)
    - Mutual Information
    - SSIM
    - Qw, Qe
    - VIF
    - CC
    - N
    
- [ ] Training with flirADAS
    - RGB + IR
    - (Y + IR) + CbCr
    - HSV color space + IR ???
    
- [ ] Dataset results 
    - TNO dataset images or 21 images and their results
    
- [ ] Fusion Layer effect
    - Add
    - Conv 1x1
    
- [ ] Effect of loss function 
    - Q
    - QE
    - MSE
    
- [ ] SR with fusion (last layer SR)

