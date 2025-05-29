## Lecture 3

### Loss functions

$$ L = \frac{1}{N} \sum_i L_i (f(x_i,W), y_i) $$ 

for a dataset ${x_i, y_i}_{i=1}^N$, 

---

#### Multi-class SVM loss

- scores vector = $f(x_i, W)$
- $ L_i = \sum_{j\ne y_i} max(0, s_j - s_{y_i} + 1) $ also known as Hinge Loss
- It only cares about whether the scores for the wrong classes is not as large as the correct class - 1.

---

#### Regularization

- L2 = $W^2$
- L1 = $|W|$
- Elastic net (L1 + L2) = $\beta W^2 + |W|$
- Max norm
- Dropout
- Batch Normalization
- Stochastic Depth

---

#### Softmax and Negative Log Likelihood loss

- $ P(Y = k | X = x_i) = \frac{e^s k}{\sum_j e^s j} $
- In softmax we convert the logits into positive numbers by exponentiating them and then we divide by the sum of all exponents to normalize everything. We get a probability distribution between 0 and 1.
- The loss then becomes, $ L_i = -log(\frac{e^s k}{\sum_j e^s j})
- We use log because it's easier for computers to work with and since log likelihood measures goodness of something, we mulitply by -1 to get a loss function.

---

### Image Features

- Some datasets are difficult for linear classifiers to work on, for example, multi-modal points. So we convert the data into a feature space where it is easier to draw lines across clear boundaries between classes.
- For images, think of sending the color histogram as a feature input or a histogram of oriendted gradients. 
