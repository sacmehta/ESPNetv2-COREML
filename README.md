# Real-time semantic segmentation using ESPNetv2 on iPhone

This repository provides **a real-time demo** of [ESPNetv2](https://arxiv.org/abs/1811.11431) on iPhone (tested only on iPhone7). Below are some illustrations.

<table>
    <tr>
        <td colspan=2 align="center"><b>Real-time semantic segmentation using ESPNetv2 on iPhone7<b></td>
    </tr>
    <tr>
        <td>
            <img src="https://github.com/sacmehta/EdgeNets/blob/master/images/espnetv2_iphone7_video_1.gif?raw=true" alt="Seg demo on iPhone7"></img>
        </td>
        <td>
            <img src="https://github.com/sacmehta/EdgeNets/blob/master/images/espnetv2_iphone7_video_2.gif?raw=true" alt="Seg demo on iPhone7"></img>
        </td>
    </tr>
</table>

## Model details
The COREML model takes an RGB image of size 256x256 as an input and produces an output of size 256x256 in **real-tim**. The model learns about `0.79 million parameters` and performs roughly `337 million FLOPs` to generate the segmentation mask. The model is trained using PyTorch on the PASCAL VOC 2012 dataset and achieves a segmentation score of `63.36`, which is measured in terms of mean interesection over union (mIOU). 

Several pre-trained models are provided in our [EdgeNets](https://github.com/sacmehta/EdgeNets) repository. 

## Contributions
If you are familiar with iOS application development and wants to improve the design or contribute in some way, please do so by creating a `pull request`. We welcome contributions.

## License
The code and models are released under the same license as [EdgeNets](https://github.com/sacmehta/EdgeNets).
