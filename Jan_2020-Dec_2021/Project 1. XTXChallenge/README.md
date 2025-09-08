######
- In this project, you will gave a dataset of stock_movement, denoted by y.
- Anothers columns be the dependent variables which be divided into 4 groups of 15 features:

        Group 1. Ask_Size
        Group 2. Bid_Size
        Group 3. Ask_Rate
        Group 4. Bid_Rate
- This problem is about the time-series forecasting; and unfortunately, we didn't have any infomation on the time-index.
- I had suggested some method to predict the value y; such as `XGB`, `LSTM`,...
- In the last step, integrate this to Lambda-function `AWS`, S3 and SSM to complete this totally

#==================================================================================
You will have 
- `requirement.txt`: python libraries to install
- a `notebook` folders included: [EDA](https://github.com/NhanDoV/All-of-my-projects./blob/main/Jan2020-Jan2022/Project%201.%20XTXChallenge/notebook/EDA.ipynb) and [feature-selection-xgb.ipynb](https://github.com/NhanDoV/All-of-my-projects./blob/main/Jan2020-Jan2022/Project%201.%20XTXChallenge/notebook/feature-selection-xgb.ipynb) to see how the model work
- and the other 4 `python scripts`

          train.py : to train model
          predict.py : to submit the result
          mantainance.py: Supposed that you are using any Cloud Service like AWS, if we have a real-time data then how can we observe and maintainance our model?
