# end2end_AU_speech
Code for the paper: 
Hai X. Pham, Y. Wang and V. Pavlovic, "End-to-end Learning for 3D Facial Animation from Speech", ICMI 2018.

Note that the learning rates in this code are different from ones used in the paper, resulting in better performance. The new results will be included in our journal version.

Our code is based on Python 3.6 and uses the following libraries:
- CNTK for deep learning
- librosa for speech processing
- pyglet for 3D rendering
- opencv
- other libraries are included in Anaconda distribution.
---
1. Train the model with provided CTF files

We generate CNTK data text format files to speed up training. You can download them from the following links:
* To train CNN model: [link](https://1drv.ms/u/s!AsfrZCEaosem8NAhY6l-CLCc8sxOlw?e=f9sOC9)
* To train recurrent models: [link](https://1drv.ms/u/s!AsfrZCEaosem8NAix81eOtEpVG_VAA?e=T2gVMF)

After that, you must edit proper paths in main() in train_end2end.py, and run the following line:

> python train_end2end.py [model_type]

where model_type can be either: --cnn / --gru / --lstm / --bigru / --bilstm. There are other options (pls discover them in the code), feel free to adjust them to your liking.

2. Evaluate a model:

Edit test_one_seq() in eval_speech.py with proper paths, then execute it. Generated video frames will be stored in the specified folder.

3. Prepare data from scratch:

* run extract_spectrogram.py to retrieve spectrograms. Again, please edit proper paths in this script.
* download AU labels from here: [link](https://1drv.ms/u/s!AsfrZCEaosem8NAgfMTpfIKR8lKLfg?e=2Bw0pD)
* run create_spectrogram_CTF.py to create CTF files. Remember to edit paths.

If you find this repository helpful, please cite our paper:

@inproceedings{  
author = {Hai X. Pham and Yuting Wang and Vladimir Pavlovic},  
title = {End-to-end Learning for 3D Facial Animation from Speech},  
booktitle = {International Conference on Multimodal Interaction},  
year = {2018}  
}
