
# Time Series Analysis Guide: Illness Dataset Example

This guide outlines the process of analyzing time series data, using the national illness dataset as an example. We'll cover various visualization and statistical techniques to gain insights from the data.

## 1. Data Preparation

First, load and prepare your time series data:

```python
import pandas as pd

# Load the dataset
illness = pd.read_csv('national_illness.csv', parse_dates=['date'])
illness.set_index('date', inplace=True)
```

## 2. Time Plots

Visualize the trends of your time series data:

```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_time_series(df, columns, layout=(2, 2), figsize=(20, 12), suptitle='Trends Over Time'):
    rows, cols = layout
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.flatten()
    
    for i, column in enumerate(columns):
        if i >= len(axs):
            break
        df[column].plot(ax=axs[i], title=f'Trends of {column} Over Time')
        axs[i].set_ylabel(column)
        axs[i].grid(True)
    
    for ax in axs[len(columns):]:
        ax.set_visible(False)
    
    plt.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

plot_time_series(illness, 
                 ['% WEIGHTED ILI', '%UNWEIGHTED ILI', 'AGE 0-4', 'AGE 5-24', 'ILITOTAL'],
                 layout=(3, 3))
```

This allows you to observe overall trends, seasonality, and potential outliers in your data.

## 3. Seasonal Plots

Analyze seasonal patterns in your data:

```python
import seaborn as sns
import numpy as np

def plot_seasonal_trends_with_legends(df, columns, layout=(3, 2), period='DayOfYear', cmap='viridis'):
    rows, cols = layout
    fig, axs = plt.subplots(rows, cols, figsize=(cols*7, rows*5))
    
    df['Year'] = df.index.year
    if period == 'DayOfYear':
        df['Period'] = df.index.dayofyear
    elif period == 'WeekOfYear':
        df['Period'] = df.index.isocalendar().week
    elif period == 'Month':
        df['Period'] = df.index.month
    
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]
    
    for i, column in enumerate(columns):
        if i < len(axs):
            ax = axs[i]
            years = df['Year'].unique()
            color_palette = sns.color_palette(cmap, n_colors=len(years))
            
            for j, year in enumerate(sorted(years)):
                yearly_data = df[df['Year'] == year]
                ax.plot(yearly_data['Period'], yearly_data[column], label=year, color=color_palette[j])
            
            ax.set_title(f'{column}')
            ax.set_xlabel('Period')
            ax.set_ylabel(column)
    
    for i in range(len(columns), len(axs)):
        axs[i].axis('off')
    
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1), title='Year')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

plot_seasonal_trends_with_legends(illness, 
                                  ['% WEIGHTED ILI', '%UNWEIGHTED ILI', 'AGE 0-4', 'AGE 5-24', 'ILITOTAL'],
                                  layout=(3, 2))
```

Seasonal plots help identify recurring patterns within specific time frames (e.g., yearly cycles).

## 4. Correlation Analysis

### 4.1 Heatmap of Correlation Matrix

Visualize relationships between variables:

```python
import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = illness.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Correlation Matrix')
plt.show()
```

### 4.2 Segmented Correlation Analysis

Analyze how correlations change over time:

```python
total_segments = 12
segment_size = illness.shape[0] // total_segments

fig, axes = plt.subplots(4, 3, figsize=(28, 28))
fig.suptitle('Segmented Data Correlation Matrix Heatmaps', fontsize=16)

for segment in range(total_segments):
    start_row = segment * segment_size
    end_row = start_row + segment_size if segment < total_segments - 1 else illness.shape[0]
    segment_df = illness.iloc[start_row:end_row].select_dtypes(include=[np.number])
    correlation_matrix = segment_df.corr()
    
    ax = axes[segment // 3, segment % 3]
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title(f'Segment {segment + 1}')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```

This technique helps identify evolving relationships between variables across different time periods.

## 5. Scatter Plots

Use pair plots to visualize relationships between variables:

```python
sns.pairplot(illness, plot_kws=dict(linewidth=0, s=4), corner=True, height=1.25)
plt.tight_layout()
plt.show()
```

Pair plots provide a quick overview of bivariate relationships and distributions.

## 6. Lag Plots

Examine autocorrelation and seasonality:

```python
from pandas.plotting import lag_plot

fig, axes = plt.subplots(3, 3, figsize=(20, 20))
axes = axes.flatten()

for i, col in enumerate(illness.columns):
    if i < len(axes):
        lag_plot(illness[col], ax=axes[i])
        axes[i].set_title(f'Lag Plot of {col}')

plt.tight_layout()
plt.show()
```

Lag plots help identify autocorrelation and potential seasonality in individual variables.


## 7. Distribution Analysis: Training vs Test Data

### 7.1 Time Series and ACF Plots

Compare the distribution of data between training and test sets using time series plots and Autocorrelation Function (ACF) plots:

```python
import numpy as np
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt

def plot_series_and_acf(data, column_name):
    train_size = int(len(data) * 0.7)
    train, test = np.split(data, [train_size])

    train_acf = acf(train[column_name], nlags=30)
    test_acf = acf(test[column_name], nlags=30)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    
    ax[0].plot(train.index, train[column_name], color='black', label='Train')
    ax[0].plot(test.index, test[column_name], color='purple', label='Test')
    ax[0].legend()
    ax[0].set_title(f'Series ({column_name})')
    
    ax[1].bar(range(len(train_acf)), train_acf, color='black')
    ax[1].set_title('Train ACF')
    
    ax[2].bar(range(len(test_acf)), test_acf, color='black')
    ax[2].set_title('Test ACF')

    plt.tight_layout()
    plt.show()

for column in illness.columns:
    plot_series_and_acf(illness, column)
```

This analysis provides three key visualizations for each variable:
1. Time Series Plot: Shows the original data split into training and test sets.
2. Training ACF Plot: Displays the autocorrelation function for the training data.
3. Test ACF Plot: Shows the autocorrelation function for the test data.

### 7.2 ACF Difference Analysis

Calculate and visualize the difference in ACF between training and test data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

# Load the dataset
file_path = 'all_six_datasets/illness/national_illness.csv'  # Replace with the actual path of your dataset
data = pd.read_csv(file_path, parse_dates=[0], index_col=0)

# Split the dataset into training and testing sets in chronological order, adjusted to 7:3 ratio
train_size = int(len(data) * 0.7)  # Now 70% of the data is used for training
test_size = len(data) - train_size  # The remaining 30% for testing
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Calculate the ACF difference for each channel
diff_acf = []
for col in data.columns:
    acf_train = acf(train_data[col], nlags=40)
    acf_test = acf(test_data[col], nlags=40)
    # Calculate the squared difference at each time lag
    diff_acf_col = np.square(acf_train - acf_test)
    # Sum the differences across all time lags for each channel, resulting in Diff_c
    diff_acf.append(np.sum(diff_acf_col))

# Sum the Diff_c for all channels, resulting in Diff_sum, and consider the normalization factor 1/C
diff_sum = np.sum(diff_acf) * (1 / len(data.columns))

# Calculate and sort the ACF differences for each channel
diff_acf_sorted = sorted([(col, diff) for col, diff in zip(data.columns, diff_acf)], key=lambda x: x[1], reverse=True)

# Unpack the sorted channels and their differences
channels, sorted_diff_acf = zip(*diff_acf_sorted)

# Plot the ACF difference for each channel
plt.figure(figsize=(12, 6))
plt.bar(channels, sorted_diff_acf, color='orange')
plt.axhline(y=diff_sum, color='red', linestyle='--', label=f'Sum Diff: {diff_sum:.4f}')

# Calculate the percentage of channels below the average
below_average = len([diff for diff in sorted_diff_acf if diff < diff_sum])
percent_below = (below_average / len(sorted_diff_acf)) * 100
plt.text(1, diff_sum, f"<{percent_below:.2f}% ({below_average}/{len(sorted_diff_acf)}) channels", va='bottom', ha='left', color='red')

plt.legend()
plt.xlabel('Channel Index')
plt.ylabel('ACF Diff')
plt.title('Difference of ACF between Training Data and Test Data')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
```

This analysis calculates and visualizes the difference in Autocorrelation Function (ACF) between the training and test data for each channel. The key components of this analysis are:

1. ACF Difference Calculation: For each channel, we compute the squared difference between the ACF of the training data and the ACF of the test data.
2. Normalization: We sum these differences and normalize by the number of channels to get an overall measure of ACF difference.
3. Visualization: The plot shows the ACF difference for each channel, sorted in descending order.
4. Average Line: A red dashed line indicates the average ACF difference across all channels.
5. Channel Distribution: The text annotation shows the percentage and number of channels below the average difference.

### Interpreting the Results:

- Similar patterns in ACF plots suggest consistent temporal dependencies and seasonal patterns.
- Differences might indicate changes in the underlying process over time or potential overfitting.
- Regular spikes in ACF plots can indicate seasonality.
- A slowly decaying ACF can suggest the presence of a trend.
- Channels with high ACF differences may require special attention in modeling, as they show more discrepancy between training and test data.
- A high percentage of channels below the average difference suggests that most channels have relatively consistent ACF patterns between training and test data.

This analysis helps in identifying which channels (variables) show the most significant differences in their autocorrelation structure between the training and test sets, which can be crucial for model selection and evaluation in time series forecasting.



## 8. White Noise Analysis

Generating and plotting white noise for each column in the dataset:

```python
import numpy as np
import matplotlib.pyplot as plt

# Assuming data_numeric is your dataset with numeric columns
fig, axs = plt.subplots(len(data_numeric.columns), 1, figsize=(10, 5*len(data_numeric.columns)))

for i, column in enumerate(data_numeric.columns):
    # Generating white noise for each column
    white_noise = np.random.normal(size=len(data_numeric[column]))
    
    # Plotting
    axs[i].plot(white_noise)
    axs[i].set_title(column)
    axs[i].set_xlabel('Index')
    axs[i].set_ylabel('White Noise')

plt.tight_layout()
plt.show()
```

This analysis generates and plots white noise for each column in the dataset. White noise is a random signal with a constant power spectral density. By comparing these white noise plots with your actual data, you can visually assess how much your data deviates from pure randomness.

## 9. Granger Causality Test

Performing Granger Causality Test on the dataset:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests

# Load the data
df = pd.read_csv('all_six_datasets/illness/national_illness.csv')
if 'date' in df.columns:
    df = df.drop('date', axis=1)
df = df.select_dtypes(include=[np.number])

max_lag = 4 # You can set the max lag that you want to test for
test = 'ssr_chi2test'  # Example test, you can also use 'lrtest', 'params_ftest', 'ssr_ftest'

results = pd.DataFrame(np.zeros((df.shape[1], df.shape[1])), columns=df.columns, index=df.columns)

for i in df.columns:
    for j in df.columns:
        if i != j:  # Avoid testing the correlation of the feature with itself
            test_result = grangercausalitytests(df[[i, j]], max_lag, verbose=False)
            min_p_value = np.min([test_result[k+1][0][test][1] for k in range(max_lag)])
            results.loc[i, j] = min_p_value

# Print the results
print(results)

# Replace any diagonal elements with NaN to avoid self-comparison confusion
np.fill_diagonal(results.values, np.nan)

# Generate a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(results, annot=True, fmt=".2f", cmap="YlGnBu", mask=np.isnan(results), cbar_kws={'label': 'p-value'})
plt.title("Granger Causality Test Results")
plt.xlabel("Variable Feature")
plt.ylabel("Variable Feature")
plt.show()
```

The Granger Causality Test is used to determine whether one time series is useful in forecasting another. This analysis performs the test for each pair of variables in the dataset and visualizes the results as a heatmap.

Interpretation:
- Lower p-values (darker colors in the heatmap) indicate stronger evidence of Granger causality.
- The heatmap shows the minimum p-value across all tested lags for each pair of variables.
- This can help identify potential causal relationships between variables in your time series data.

## 10. Probability Density Plots

Generating probability density plots for each column, comparing training and testing data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Assuming df is your dataset and train_end_index is the index where training data ends
plots_per_row = 3
num_rows = (len(df.columns) + plots_per_row - 1) // plots_per_row

plt.figure(figsize=(plots_per_row * 7, num_rows * 6))

for i, column in enumerate(df.columns):
    plt.subplot(num_rows, plots_per_row, i + 1)
    # Split data into train and test
    train_data = df[column][:train_end_index]
    test_data = df[column][train_end_index:]  # Start the test data right after the train data
    
    # Training data histogram and PDF
    plt.hist(train_data, bins=30, density=True, alpha=0.5, color='b', edgecolor='black', label='Train')
    mu_train, std_train = norm.fit(train_data)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p_train = norm.pdf(x, mu_train, std_train)
    plt.plot(x, p_train, 'b', linewidth=2)
    
    # Test data histogram and PDF
    plt.hist(test_data, bins=30, density=True, alpha=0.5, color='r', edgecolor='black', label='Test')
    mu_test, std_test = norm.fit(test_data)
    p_test = norm.pdf(x, mu_test, std_test)
    plt.plot(x, p_test, 'r', linewidth=2)
    
    title = f"{column}: Train (mu = {mu_train:.2f}, std = {std_train:.2f}), Test (mu = {mu_test:.2f}, std = {std_test:.2f})"
    plt.title(title)
    plt.legend()

# Adjust layout
plt.tight_layout()
plt.show()
```

This analysis creates probability density plots for each column in the dataset, comparing the distribution of the training and testing data.

Interpretation:
- Blue represents the training data, and red represents the testing data.
- Purple areas indicate overlap between training and testing distributions.
- The plots show both histograms and fitted normal distributions for each dataset.
- Mean (mu) and standard deviation (std) are provided for both training and testing data.
- These plots help visualize how well the distribution of the test data matches that of the training data, which is crucial for assessing the generalizability of your models.

By comparing the distributions, you can identify potential issues such as:
1. Shift in distribution between training and test data
2. Differences in variability between training and test data
3. Presence of outliers or anomalies in either dataset

These insights can guide your modeling approach and help in identifying potential challenges in applying your model to new data.

## Conclusion

By applying these techniques to your time series data, you can:

1. Identify overall trends and patterns
2. Detect seasonality and cyclical behavior
3. Understand relationships between variables
4. Spot potential outliers or anomalies
5. Examine how variables correlate with their past values
6. Ensure consistency between training and test datasets

These insights form the foundation for further analysis, such as forecasting or anomaly detection in time series data.
