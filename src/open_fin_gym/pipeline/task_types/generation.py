from .base import TaskTypeParams

GenerationParams = TaskTypeParams(
    name="generation",
    example="""
## Example: generation

If the paper describes an experiment where they fit a generative model to daily equity returns and assessed the realism of sampled
paths against held-out real returns, the specification could include:

- Description: The task is to implement a generative model of daily log-returns, trained on a [M, 250] matrix of real return windows.
  You should sample a [N, 250] matrix of synthetic windows and write it out. Sampling is unconditional, so there are no test inputs.
  Your output is assessed distributionally against a held-out sample of real windows.
- Training input data:
    - returns_train a [M, 250] matrix of M 250-day AAPL log-return windows from 2015-2019
- Training target data:
    - (none, the model is fit to the training inputs)
- Test input data:
    - (none, sampling is unconditional)
- Test output data:
    - samples a [N, 250] matrix of N synthetic 250-day log-return windows
- Test target data:
    - returns_test a [K, 250] matrix of held-out real 250-day log-return windows from 2020
- Assessment metrics:
    - wasserstein: 1-Wasserstein distance between the marginal return distributions of samples and returns_test
    - acf_error: Absolute error between the autocorrelation functions of the squared returns of samples and returns_test
""".strip(),
    required_groups={
        "training_inputs",
        "test_targets",
        "test_outputs",
    },
    row_correspondence=(
        "The user samples their output, so compare the two datasets as "
        "distributions and do not assume the rows are paired."
    ),
)
