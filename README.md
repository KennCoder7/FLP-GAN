# [FLP-GAN](https://kenncoder7.github.io/2025/01/30/FLPGAN/)
This is an official PyTorch implementation of "Text-to-Face Synthesis based on Facial Landmarks Prediction" in Machine Vision and Applications.

Utilzing AI models to generate faical image based on textual descriptions is an interest thing in AICG filed. However, directly generating images from text conditional inputs is diffcult because of domain gap. Thus, we proposed the FLP-GAN models, which leverages facial landmark as a semantic bridge to faciliate the generation from text to facial images.

![](https://github.com/KennCoder7/KennCoder7.github.io/blob/master/assets/flpgan1.png)

## Paper Info
- **Title**: Text-to-face synthesis based on facial landmarks prediction
- **Author(s)**: K Wang, L Chen, B Cao, B Liu, J Cao 
- **Conference/Journal Name**: Machine Vision and Applications
- **Date**: 2025
- **Link**: [Springer](https://link.springer.com/article/10.1007/s00138-024-01624-1)
- **GitHub**: [FLP-GAN](https://github.com/KennCoder7/FLP-GAN)
  
## Abstract
The human face, being one of the most prominent physical features, plays a crucial role in appearance description and recognition. Consequently, text-to-face synthesis has garnered increasing interest in the research community, with applications in criminal investigation, image editing, and more. Compared to text-to-image synthesis, generating facial images from text requires more specialized knowledge due to the subjectivity and diversity of facial descriptions, which involve more fine-grained appearance features. In this paper, we propose a text-to-face synthesis model based on Facial Landmarks Prediction (FLP-GAN). Specifically, we design two foundational submodules to facilitate the generation task. First, a co-attention mechanism is employed to pretrain the image and text encoders to extract features related to facial information. Second, a facial landmarks prediction model is proposed to generate face segment maps based on descriptive text, providing facial semantic prior knowledge for the subsequent face synthesis process. Conditioned on the semantic features obtained from the submodules, we construct the text-to-face synthesis model, which incorporates a memory network and a segment fuse layer to highlight important text information and refine the features. Additionally, a multi-stage refinement process is designed to generate high-resolution face images. Experimental results on the Face2Text dataset demonstrate that our FLP-GAN model outperforms the state-of-the-art methods in both qualitative and quantitative evaluations. Specifically, our model achieved a 22.7% improvement in Fréchet FaceNet Distence compared to the SOTA models.

## Citation
If you find this idea helpful, please consider citing:
```
@article{wang2025text,
  title={Text-to-face synthesis based on facial landmarks prediction},
  author={Wang, Kun and Chen, Lei and Cao, Biwei and Liu, Bo and Cao, Jiuxin},
  journal={Machine Vision and Applications},
  volume={36},
  number={1},
  pages={1--17},
  year={2025},
  publisher={Springer}
}
```
