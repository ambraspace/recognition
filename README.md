# INFO

This is an application of "[A Convolutional Neural Network Cascade for Face Detection](https://openaccess.thecvf.com/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf)" - academic paper by Haoxiang Li, Zhe Lin, Xiaohui Shen, Jonathan Brandt and Gang Hua.

The author used AFLW dataset which couldn't be used for commercial applications, so I used LFW dataset in combination with OpenCV face feature detection.

First, LFW images are turned to grayscale to speed-up detection on mini PCs (Raspberry Pi). After that, I manually removed all false detections. Also,
some images contained more then one detected face, so only one has been kept. After that, all images have been mirrored to get double amount of images.

The success rate achieved was over 93% for face detection and over 88% for gender classification, although in real world applications the success rate was
noticeably lower.

![2002-07-19-big-img_135](https://github.com/user-attachments/assets/df9762ae-f85e-4400-a4de-c762fa262c3c)

![2002-07-20-big-img_588](https://github.com/user-attachments/assets/0aec8abf-fdf6-4512-84f3-7078c169862d)

## LICENSE

None.
