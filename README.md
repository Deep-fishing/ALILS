# A-Learnable-Image-Based-Load-Signature-Construction-Approach-in-NILM-for-Appliances-Identification
## Abstract
One of the tasks of Non-Intrusive Load Monitoring (NILM) is load identification, which aims to extract and classify altered electrical signals after switching events are detected. In this subtask, representative and distinguishable load signatures are essential. At present, the literature approach to characterize electrical appliances is mainly based on manual feature engineering. However, the performance of signatures obtained by this way is limited. In this paper, we propose a novel load signature construction method utilizing deep learning techniques. Specifically, three learnable load signatures are presented such as Learnable Recurrent Graph (LRG), Learnable Gramian Matrix (LGM) and Generative Graph (GG). Furthermore, we test different frameworks for learning these signatures and conclude that Temporal Convolutional Networks (TCN) based on residual learning are more suitable for this work than the other schemes mentioned. The results of experiment on the PLAID datasets with submetered and aggregated, WHITED dataset and LILAC dataset confirm that our method outperforms the voltage-current trajectory, Recursive Graph and Gramian Angular Field methods in multiple evaluation metrics. 
## Data Preparation
The format of the data folder is as follows：



1 data
2  |-lilac
3      |-aggregated
4             |-current.npy
5             |-labels.npy
6             |-voltage.npy
7  |-plaid2018
8      |-aggregated
9             |-current.npy
10             |-labels.npy
11             |-voltage.npy
12      |-sub
13         |-plaid2018_sub.pickle
14  |-whited
15       |-whited.pickle

