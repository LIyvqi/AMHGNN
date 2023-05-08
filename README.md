# AMHGNN

This is the code for paper AMHGNN, our work is currently under review. The source codes and the implementation details are submitted as supplementary material to the conference site. We will open the source once our paper is accepted.


## The Description of Datasets
- **Yelp** \cite{DBLP:conf/kdd/RayanaA15}: This dataset shows review information for hotels and restaurants on Yelp. The vertices represent reviews, and the edges have three different types of relationships, such as reviews posted by the same user and reviews posted under the same product with the same rating or month.
- **Amazon** \cite{DBLP:conf/www/McAuleyL13}: This dataset is a user review network for the Musical Instruments category on Amazon. The vertices represent users, and the edges have three different types of relationships between users, such as reviewing the same product, having the same star rating within one week, or having top-5% mutual review similarities.
- **DBLP** \cite{lv2021we}: This dataset is a citation network for computer science. There are four types of vertices: authors, papers, terms, and venues. For the classification task, we choose the author vertices as the primary type.
- **ACM** \cite{DBLP:conf/www/WangJSWYCY19}: This citation network also has four types of vertices. We preserve all edges, including paper citations and references, in line with HGB \cite{lv2021we}.

## Environments

Operating system:  Ubuntu 18.04.5

CPU: Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz

GPU: NVIDIA GeForce RTX 3090

python == 3.6

## Requirements
``` bash
dgl == 0.9.1

sklearn == 0.24.2

torch == 1.9.1+cu111
```

## Train and Evaluate

``` bash
python train.py
```
