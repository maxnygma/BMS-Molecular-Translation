# **BMS-Molecular-Translation**

## **Introduction**
This is a pipeline for [Bristol-Myers Squibb – Molecular Translation](https://www.kaggle.com/c/bms-molecular-translation) by 
[Vadim Timakin](https://github.com/t0efL) and [Maksim Zhdanov](https://github.com/xzcodes). We got bronze medals in this competition.
Significant part of code was originated from 
[Y.Nakama's notebook](https://www.kaggle.com/yasufuminakama/inchi-resnet-lstm-with-attention-starter) 

This competition was about image-to-text translation of images with molecular skeletal strucutures to InChI chemical formula identifiers.

![](https://www.kaggleusercontent.com/kf/56177275/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..KdFGELi4smozCcC_cPCdgg.yiauS_UdZbed07asLdsvH8I3ACfE-6mDp_GyNrG-YF_CIWXu_XZDRJEBYtph-7k84wbkPdpIOB0WqUx5sFo38nR3U-SU3ZiYJqxfg3WA-vkqZzwHbqK4-yAFUAErVRFkIMgqo4cSAFoQ6uUQld-HkLR2ErgL7tLDE8KqwaFtpocpNgpspeqjSUoP0P5qqABGJOtDgj2mt-SiwQ9IHm29kQBaYHTTS3yGsf_6kb97XfhLYiyYjg2C2ITuVA75hzrVG_d7TWAlM2oe2v1U8a0OiioRk90IRQcdNwOreRnNnNc42Q4KmZe1budXxoboxZOOAkr8JlG4fAVstbm5YDkYOOJNFh9hRvp_ytOPnS9ljCc-yMeX5J82enQTRTWRvg1ahmMZSuAle51_WBn9eRdvysq7FIUDq66nbwrbUGzydVbBWNtzjWELMwscPUF349VdVpc7r16GtAnwVHl2-4EbqSkJI1Hit-_LdfWVquCG2xAAK-8xdcWaCnpDEymRwcvyndyAn22Unz-ZPjT-6VVvuqvRpSh-7rciOYFfiW8nxGj7ED3fhDrg1dkr097yabWPjS2urRxQinauotAX9D_GvglkV5NKlbyjeHlJvJUfI4PBxTlwdYccGAZ9FB6SG_cXs8CBu0T8eOfVNA3JElRZvDDSQymr0Acg6ZIoadj6E8RDG2G_9tIPZDwGJ0WmS2FQ._bwqbST0RSYL68hqfLSCsQ/__results___files/__results___12_1.png)
<br><br>
**InChI=1S/C16H13Cl2NO3/c1-10-2-4-11(5-3-10)16(21)22-9-15(20)19-14-8-12(17)6-7-13(14)18/h2-8H,9H2,1H3,(H,19,20)**

## **Solution**

### **General Encoder-Decoder concept**
Most participants used CNN encoder to acquire features with decoder (LSTM/GRU/Transformer) to get text sequences. 
That's a casual approach to image captioning problem.
<br><br>
<p align="center">
  <img src="https://raw.githubusercontent.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/master/img/decoder_att.png" width="700" align="center">
</p>
 
### **Pseudo-labelling with InChI validation using RDKit**
RDKit is an open source toolkit for cheminformatics and it was quite useful while solving the problem. When we trained our first model, it scored around 7-8
on public leaderboard and we decided to make pseudo-labelling on test data. However, in common scenario you get a significant amount of wrong predictions in your
extended training set from pseudo-labelling. With RDKit we validated all of our predicted formulas and select around 800k correct samples. Lack of wrong labels
in pseudo labels improved the score.

### Predictions normalization

[This notebook tells about InChI normalization](https://www.kaggle.com/nofreewill/normalize-your-predictions)

### Blending

Finally, we blended ~20 predictions from 2 models (mostly from different epochs) using RDKit validation to choose only 
formulas which have possible InChI structure.

#### **Final private LB score 1.79**
