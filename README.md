# dual_stage_attention_rnn

In an attempt to learn Tensorflow, I have implemented the model in 
[A Dual-Stage Attention-Based Recurrent Neural Network
for Time Series Prediction](https://arxiv.org/pdf/1704.02971.pdf)
using Tensorflow 1.13.
- Nasdaq data is used for testing, which is from repo [da-rnn](https://github.com/Seanny123/da-rnn/blob/master/data/).
- Based on the [discussion](https://github.com/Seanny123/da-rnn/issues/4), i implemented both cases where current 
exogenous factor is included, i.e.,
<img src="https://latex.codecogs.com/gif.latex?\widehat{y}_{T}=f(y_1,y_2,...,y_{T-1},x_1,x_2,...,x_T)" title="\widehat{y}_{T}=f(y_1,y_2,...,y_{T-1},x_1,x_2,...,x_T)" /></a>
as well as excluded, i.e. <img src="https://latex.codecogs.com/gif.latex?\widehat{y}_{T}=f(y_1,y_2,...,y_{T-1},x_1,x_2,...,x_{T-1})" title="\widehat{y}_{T}=f(y_1,y_2,...,y_{T-1},x_1,x_2,...,x_{T-1})" /> .
The switch between the two modes is control by **FLAGS.use_cur_exg** in *da_rnn/main.py*.
- To avoid overfitting, a flag to shuffle the train data has been added, activated by **FLAGS.shuffle_train** in *da_rnn/main.py*.
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

Suppose we want to run 200 epochs and use Tensorboard to 
visualize the process

```bash
cd da_rnn
python main.py --write_summary True --max_epoch 200
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
    
   
Results of my experiments are list below. Running more epochs and use larger encoder/decoder dimension could possibly 
achieve better results.
     
| # Epoch | Shuffle Train | Use Current Exg| Econder/Decoder Dim | RMSE |  MAE| MAPE  |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | False |  False  | 32     | 105.671| 104.60 | 2.15%|
| 100 | True |  False  |    32  |29.849 | 29.033 |0.59% | 
| 100 | False |  True  | 32     | 46.287| 32.398 |0.66% |
| 100 | True |  True  |    32  |1.491 | 1.172 | 0.024%|


```bash
# To shuffle the train data and use current exogeneous factor
python main.py --write_summary True --max_epoch 100 --shuffle_train True --use_cur_exg True
```

After 100 epochs(with data shuffled and current exogenous factors used) the prediction is plot as     


<img src="https://github.com/iLampard/dual_stage_attention_rnn/blob/master/figures/pred_plot.png" />     
     
## Requirement

```bash
tensorflow==1.13.1
scikit-learn==0.21.3
numpy==1.16.4
```

Although I have not tested, I guess it should be working under tf 1.12 and tf 1.14 as well.

# Reference
- [A PyTorch Example to Use RNN for Financial Prediction](http://chandlerzuo.github.io/blog/2017/11/darnn)
- [Github: da-rnn](https://github.com/Seanny123/da-rnn)