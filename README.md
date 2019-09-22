# dual_stage_attention_rnn

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

In an attempt to learn Tensorflow, I have implemented the model in 
[A Dual-Stage Attention-Based Recurrent Neural Network
for Time Series Prediction](https://arxiv.org/pdf/1704.02971.pdf)
using Tensorflow 1.13.
- Nasdaq data is used for testing, which is from repo [da-rnn](https://github.com/Seanny123/da-rnn/blob/master/data/).
- Based on the [discussion](https://github.com/Seanny123/da-rnn/issues/4), i implemented both cases where current 
exogenous factor is included, i.e.,
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{y}_T=f(y_1,..y_{T-1}, x_1,x_2,...,x_T)" /> 
as well as excluded, i.e. <img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{y}_T=f(y_1,..y_{T-1}, x_1,x_2,...,x_{T-1})" /> .
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

Suppose we want to run 500 epochs and use Tensorboard to 
visualize the process

```bash
cd da_rnn
python main.py --write_summary True --max_epoch 500
```

To check the description of all flags
```bash
python main.py -helpful
```

To open tensorboard
```bash
tensorboard --logdir=path
```

where *path* can be found in the log which shows the relative dir where the model is saved, e.g. 
*logs/ModelWrapper/lr-0.001_encoder-32_decoder-32/20190922-103703/saved_model/tfb_dir*.


 
### Test result 
    
   
    
     
| # Epoch | Shuffle Train | Use Current Exg| Econder/Decoder Dim | RMSE |  MAE| MAPE  |
| --- | --- | --- | --- | --- | --- | --- |
| 500 | False |  False  | 32     | |  |
| 500 | True |  False  |    32  | |  |
| 500 | False |  True  | 32     | |  |
| 500 | True |  True  |    32  | |  |


One example of command line to run the script
```bash
python main.py --write_summary True --max_epoch 500 --shuffle_train True --use_cur_exg True
```

     
## Requirement

```bash
tensorflow==1.13.1
scikit-learn
```

Although I have not tested, I guess it should be working under tf 1.12 and tf 1.14 as well.

# Reference
- [A PyTorch Example to Use RNN for Financial Prediction](http://chandlerzuo.github.io/blog/2017/11/darnn)
- [Github: da-rnn](https://github.com/Seanny123/da-rnn)