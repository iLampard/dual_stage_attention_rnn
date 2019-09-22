# dual_stage_attention_rnn

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

In an attempt to learn Tensorflow, I have implemented the model in 
[A Dual-Stage Attention-Based Recurrent Neural Network
for Time Series Prediction](https://arxiv.org/pdf/1704.02971.pdf)
using Tensorflow 1.13.
- Nasdaq data is used for testing, which is from repo [da-rnn](https://github.com/Seanny123/da-rnn/blob/master/data/).
- Based on the [discussion](https://github.com/Seanny123/da-rnn/issues/4), i implemented both cases where current 
exogenous factor is included, i.e., 
$$\hat{y}_T=f(y_1,..y_{T-1}, x_1,x_2,...,x_T)$$
as well as excluded, i.e. $$\hat{y}_T=f(y_1,..y_{T-1}, x_1,x_2,...,x_{T-1})$$.
- A ModelRunner class is added to control the pipeline of model training and evaluation.
## Run

### Source Data
Put the downloaded Nasdaq csv file under *data/data_nasdaq*.
```bash
da_rnn
|__data
    |__data_nasdaq
            |__nasdaq100_padding.csv
```

### Run the training and prediction pipeline

 
### Test result 
    
   
    
     
| # Epoch | Shuffle Train | Use Current Exg| Econder/Decoder Dim | RMSE |  MAE| MAPE  |
| --- | --- | --- | --- | --- | --- | --- |
| 500 | False |  False  | 32     | |  |
| 500 | True |  False  |    32  | |  |
| 500 | False |  True  | 32     | |  |
| 500 | True |  True  |    32  | |  |

     
## Requirement

```bash
tensorflow==1.13.1
scikit-learn
```

Although I have not tested, I guess it should be working under tf 1.12 and tf 1.14 as well.

# Reference
- [A PyTorch Example to Use RNN for Financial Prediction](http://chandlerzuo.github.io/blog/2017/11/darnn)
- [Github: da-rnn](https://github.com/Seanny123/da-rnn)