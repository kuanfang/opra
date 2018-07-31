# OPRA Dataset: Online Product Reviews for Affordances

### Introduction
The OPRA Dataset was introduced in our [Demo2Vec paper](http://ai.stanford.edu/~kuanfang/pdf/demo2vec2018cvpr) for reasoning object affordances from online demonstration videos. It contains 11,505 demonstration clips and 2,512 object images scraped from 6 popular YouTube product review channels along with the corresponding affordance annotations. More details can be found on our [website](https://sites.google.com/view/demo2vec/).

### Citation
```
@inproceedings{demo2vec2018cvpr,
  title={Demo2Vec: Reasoning Object Affordances from Online Videos},
  author={Fang, Kuan and Wu, Te-Lin and Yang, Daniel and Savarese, Silvio and Lim, Joseph J},
  booktitle={CVPR},
  year={2018}
}
```

### Requirements

Install [youtube-dl](https://github.com/rg3/youtube-dl):
```Shell
sudo -H pip install --upgrade youtube-dlkj:tabe 
```

Install [ffmpeg](https://www.ffmpeg.org/):
```Shell
sudo add-apt-repository ppa:mc3man/trusty-media  
sudo apt-get update  
sudo apt-get install ffmpeg  
```

To visualize the dataset using the Ipython notebooks, these packages need to be installed: [NumPy](https://scipy.org/install.html), [OpenCV](https://docs.opencv.org/3.4.1/d2/de6/tutorial_py_setup_in_ubuntu.html), [Pillow](https://pillow.readthedocs.io/en/latest/installation.html), [Jupyter](http://jupyter.org/install).

### Usage

Download the `data/` from this [link](ftp://cs.stanford.edu/cs/cvgl/OPRA/data.zip). The folder contains playlists of YouTube product review videos (`playlists/`), product images (`images/`), and human annotations of the video segmentation and interactiion regions (`annotations/`).
```Shell
wget ftp://cs.stanford.edu/cs/cvgl/OPRA/data.zip
unzip data.zip
```

Download the product review videos from YouTube. (Note that some of the url may be no longer valid when you run the script, because they have been deleted from the playlist or due to other technical issues.)
```Shell
python download.py --playlist data/playlists --output data/raw_videos/
```

Segment the videos according to the annotations.
```Shell
python segment.py --annotations data/annotations/train.txt --raw data/raw_videos/ --output data/clips
python segment.py --annotations data/annotations/test.txt --raw data/raw_videos/ --output data/clips
```
