from .base import TaskTypeParams

ForecastingParams = TaskTypeParams(
    name="forecasting",
    example="""
## Example: forecasting

If the paper describes an experiment where they assessed their models ability to predict 7 days of a stock prices from a 30 day window
assessed on the MSE between the predicted and ground-truth prices, the specification could include:

- Description: The task is to implement a model that predicts the following 7 days of prices of a single stock, given the previous 30 days.
  You are provided with a training set consisting of a [M, 30] matrix of input windows, and corresponding [M, 7] target outputs. You are
  also provided with a [N, 30] set of test inputs and you should produce a [N, 7] matrix of outputs for assessment against the ground truth.
  The output of your model will be assessed using the MSE between your output and the ground-truth.
- Training input data:
    - x_train a [M, 30] matrix of M 30-day AAPL stock price windows randomly sampled from 2020
- Training target data:
    - y_train a [M, 7] target price windows following on from x_train
- Test input data:
    - x_test a [N, 30] matrix of N 30-day AAPL stock price windows randomly sampled from 2020
- Test output data:
    - y_pred a [N, 7] matrix of predicted prices for the 7 days following on from the provided x_test
- Test target data:
    - y_test a [N, 7] matrix of target price windows following on from x_test
- Assessment metrics:
    - MSE: The mean-squared-error between y_pred and y_test
""".strip(),
    required_groups={
        "training_inputs",
        "training_targets",
        "test_inputs",
        "test_targets",
        "test_outputs",
    },
    row_correspondence="Each row of the user output matches the same row of the test target.",
)
