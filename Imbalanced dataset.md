### Undersampling
- n1 = 100 and n2 = 900
- Create new dataset with 100 n1 points and 100 n2 points randomly selected; result 100 n1 and 100 n2 points;
- We are discarding valuable information; 80% of the dataset is discarded

### Oversampling
- n1 = 100 and n2 = 900
- Create new dataset with 900 n1 points by repeating each point 9 times; and 900 n2 points; repeating   more points from minority class to make the dataset a balanced dataset;
- We can create artificial or synthetic new points through extrapolation to increase n1 from 100 to 900;
- We are not losing any data; we can also give weights to classes; more weight to minority class;
- The nearest data point if belongs to minority class it is counted as 9 points;

#### Note: When directly using the original imbalanced dataset; we can get high accuracy with a dumb model that predicts every query point to belong to majority class;


