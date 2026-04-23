# LSF

This is the official version of LSF (Arbitrary style transfer via learning the separation and fusion of content and style).

### Preparations

Download [vgg_normalized.pth](https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0) (It is based on project [AdaIN](https://github.com/naoto0804/pytorch-AdaIN)) and put it under models/.

Download COCO2014 dataset (content dataset) and Wikiart dataset (style dataset).

### Train

python train.py --content_dir /MSCOCO/ --style_dir /Wikiart/

### Test

To use the pre-trained models, please download here [pre-trained model](https://drive.google.com/drive/folders/1R8hAQct9RcofntPZsekoj4oD7BN4e3Fx?usp=sharing) and put it under experiments/.

python test.py --content_dir /your_content_images/ --style_dir /your_style_images/

### Note

If this code is useful to you, please cite our paper.

```
@article{yu2025arbitrary,
  title={Arbitrary style transfer via learning the separation and fusion of content and style},
  author={Yu, Xiaoming and Tian, Jie and Hu, Zhenhua},
  journal={Neurocomputing},
  pages={131965},
  year={2025},
  publisher={Elsevier}
}
```
