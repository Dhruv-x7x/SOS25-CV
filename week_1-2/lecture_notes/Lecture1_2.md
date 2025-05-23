## Lecture 1 and 2

### Classification Challenges

- **Occlusion**: when the object we want to detect/classify is hidden in the background. e.g. a cat hidden behind tall grass is occluded from view
- **Background clutter**: when there is a lot of things going on in the background that look similar to the object we want to detect/classify. e.g. a orange or yellow cat lying in dry grass which is of similar color and it is hard to make out the cat
- **Intra-class Variation**: when there are variations of the object in the same image. e.g. different colored cats in the same image

---

### Types of classifiers

- **Nearest Neighbor**
- **K-Nearest Neighbor**
- **Linear Classifier**

---

#### Nearest Neighbors

- This classifier memorizes all training data and labels and then during prediction time, it finds the most similar looking training image to the test input and returns its label.
- The loss computed to find the training example match can be various distance metrics. The most commonly used is L1 (Manhattan) distance.
  - $$L1 = d(I_1, I_2) = \sum_p |I_1^p - I_2^p|$$
  - A pixel wise difference between the images is computed.
- Time Complexity:
  - Train: O(1)
  - Test: O(n) -> This is bad because we want fast predictions. It is okay if the training takes time.

---

#### K-Nearest Neighbors

- Instead of finding the closest training example to the testing input as in Nearest Neighbor classifier, in KNN we find the 'k' closest training examples to the test input and then we make predictions based on the majority class for classification and average for regression. e.g. if the input test image was of an apple, and the 5 closest training examples were 3 apples and 2 oranges, then the majority class is apples and we return that as our prediction.
- KNN is a **non-parametric** and **instance-based** learning method:
  - **Non-parametric**: No fixed parameters. We don't have to make any assumptions about the underlying distribution of the data. e.g. when fitting a line through a data, we say that it has a slope and y-intercept, which are parameters. On the contrary, in non-parametric approaches, the model decides what 'parameters' to choose from based on the data its given.
  - **Instance-based**: Also known as lazy learning. We simply memorize during training and all computation happens at prediction time. It calculates distances to all training examples when predicting. This is useful when we don't make any assumptions about the training data.
- Find the best 'k' using cross-validation (k-fold cross validation). We divide the training data into folds and run the model on each fold, then take an average over all folds.
  - Elbow method: Plot error for a range of 'k' values. It should decrease as 'k' increases but after a certain point it starts increasing again, giving the plot a 'elbow' type of shape. The lowest point of the elbow is taken as the best 'k'.
  - Odd k: We prefer taking 'k' as odd to avoid ties.
- Distance metrics:
  - L1
  - L2 (Euclidean)
    -  $$L2 = d(I_1, I_2) = \sqrt(\sum_p (I_1^p - I_2^p )^2 )$$

---

#### Linear Classifier (Template Matching)
