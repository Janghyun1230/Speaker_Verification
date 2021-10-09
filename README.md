# Speaker_Verification
Tensorflow implementation of [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467) ([Kaggle](https://www.kaggle.com/beastlyprime/tensorflow-speaker-verification), [paperswithcode](https://paperswithcode.com/paper/end-to-end-text-dependent-speaker)). This paper is based on the previous work [End-to-End Text-Dependent Speaker Verification](https://arxiv.org/abs/1509.08062).

### Speaker Verification
- **Speaker verification** does **1-1 check** between the enrolled voice and the new voice. This task requires to achieve the higher accuracy than **speaker identification** which does **N-1 check** between the N enrolled voices and a new voice. 
- There are two types of speaker verification: 1) Text dependent speaker verification (**TD-SV**). 2) Text independent speaker verification (**TI-SV**). The former uses the text-specific utterances for enrollment and verification, whereas the latter uses text-independent utterances.
- At each forward step of the method, the utterance similarity matrix is calculated and the integrated loss is used for the objective function. (see Section 2.1 of the paper)


### Files
- *configuration.py* : Argument parsing
- *data_preprocess.py* : Extracts noise and performs STFT on raw audio. For each raw audio, the voice activity detection is performed via librosa library.
- *utils.py* : Contains various util functions for training and test.  
- *model.py* : Contains train and test functions.
- *main.py* : After the dataset is prepared, run
```
python main.py --train True --model_path where_you_want_to_save                 # training
python main.py --train False --model_path model_path used at training phase     # test
```

### Data
- Note, The authors of the paper used their own private dataset, and I could not obtain it.
- In this implementation, I used VTCK public dataset, [CSTR VCTK Corpus](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) and [noise added VTCK dataset](https://datashare.is.ed.ac.uk/handle/10283/1942) (from "Noisy speech database for training speech enhancement algorithms and TTS models").
- The VCTK dataset includes speech data uttered by 109 native English speakers with various accents. 
- For TD-SV, I used the first audio file of each speaker, which is speaking *"Call Stella"*. For the each training and test data, I added random noise extracted from the noise added VTCK dataset. 
- For TD-SI, I used randomly selected utterances from each speaker. The blanks of raw audio files are trimmed, and then slicing is performed.  


### Results
I trained the model with my notebook CPU. The model hyperpameters are following the paper:
- 3 LSTM layers with 128 hidden nodes, 64 projection nodes (Total 210434 variables)
- 0.01 lr sgd with 0.5 decay
- l2 norm clipping by 3  

To finish training and test in time, I used smaller batch (4 speakers x 5 utterances) than the paper. I used the first 85% of the dataset as training set and used the remained parts as the testset. In the below, I used softmax loss (however, the contrastive loss is also implemented in this code). On my environment, it takes less than 1s for calculating 40 utterances embedding.

1) **TD-SV**  
For each utterance, random noise is added at each forward step. I tested a model after 60000 iteration. As a result, Equal Error Rate (EER) is 0, and we can see the model performs well with a small population. 
<img src=Results/TDSV_loss.JPG width="300">

The figure below contains a similarity matrix and its EER, FAR, and FRR.
Here, each matrix corresponds to each speaker. If we call the first matrix as A (5x4), then A[i,j] means the cosine similarity between the first speaker's i^th vertification utterance and the j^th speaker's enrollment.

<img src=Results/TDSV_결과.JPG width="400">


2) **TI-SV**   
Randomly selected utterances are used. I tested the model after 60000 iteration. Here, Equal Error Rate (EER) is 0.09.  

<img src=Results/TISV_loss.JPG width="300">

<img src=Results/TISV_결과.JPG width="400">


### LICENSE
MIT License







