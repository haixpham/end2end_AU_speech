# end2end_AU_speech
Code for the paper: 
Hai X. Pham, Y. Wang and V. Pavlovic, "End-to-end Learning for 3D Facial Animation from Speech", ICMI 2018.

Note that the learning rates in this code are different from ones used in the paper, resulting in better performance. The new results will be included in our journal version.

Our code is based on Python 3.6 and uses the following libraries:
- CNTK for deep learning
- librosa for image processing
- pyglet for 3D rendering
- other libraries are included in Anaconda distribution.
---
1. Train the model with provided CTF files

We generate CNTK data text format files to speed up training. You can download them from the following links:
* To train CNN model: [link](https://drive.google.com/open?id=18nve_2P-3x0i245pEsm-npxHVtED_Zis)
* To train recurrent models: [link](https://drive.google.com/open?id=1xkoMQ7sxrDtU4oLq9VaXfWhNEigwF8zz)

After that, you must edit proper paths in main() in train_end2end.py, and run the following line:

> python train_end2end.py [model_type]

where model_type can be: --cnn / --gru / --lstm / --bigru / --bilstm. There are other options, feel free to adjust them to your liking.

2. Evaluate a model:

Edit test_one_seq() in eval_speech.py with proper paths, then execute it. Generated video frames will be stored in the specified folder.

3. Prepare data from scratch:

* run extract_spectrogram.py to retrieve spectrograms. Again, please edit proper paths in this script.
* download AU labels from here: [link](https://drive.google.com/open?id=1lhBKAHm2Vw_6MAdp6KK-uPVlXj_UBK-3)
* run create_spectrogram_CTF.py to create CTF files. Rememver to edit paths.

If you use this code in your publications, please cite our paper:

@inproceedings{  
author = {Hai X. Pham and Yuting Wang and Vladimir Pavlovic},  
title = {End-to-end Learning for 3D Facial Animation from Speech},  
booktitle = {International Conference on Multimodal Interaction},  
year = {2018}  
}
