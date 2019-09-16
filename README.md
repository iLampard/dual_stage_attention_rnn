# dual_stage_attention_rnn


In an attempt to learn Tensorflow, I have implemented the model in 
[A Dual-Stage Attention-Based Recurrent Neural Network
for Time Series Prediction](https://arxiv.org/pdf/1704.02971.pdf)
using Tensorflow 1.13.
 
 
 
## Run

### Source Data
For testing, NASDAQ data from repo [da-rnn](https://github.com/Seanny123/da-rnn/blob/master/data/) is used. 
 
 
## Requirement

```bash
tensorflow==1.13.1
```

Although I have not tested, I guess it should be working under tf 1.12 and tf 1.14 as well.

# Reference
- [A PyTorch Example to Use RNN for Financial Prediction](http://chandlerzuo.github.io/blog/2017/11/darnn)
- [Github: da-rnn](https://github.com/Seanny123/da-rnn)