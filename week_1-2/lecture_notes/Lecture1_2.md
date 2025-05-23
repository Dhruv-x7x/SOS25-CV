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

- An abstraction of what the actual machine learning models do. MLP is made from this classifier but we add the activation functions to make it non-linear.
- Suppose we have the CIFAR10 dataset. A dataset of 50000 images across 10 classes. Each image is 32x32x3 = 3072 pixels. The linear classifier does the following:
  - Image (3072 x 1) ---> f(x, W) ---> 10 Class scores
  - Max of the class scores is our prediction. W is the weights vector. It contains a template for each class and matches the input image with it.
  - To combine the weights and input, we can just multiply.
    - $f(x, W) = Wx + b$
      - here $x$ is $3071 \times 1$, $W$ is $10 \times 3072$. Sometimes when our classes are unbalanced we add a *bias* term $b$ which is $10 \times 1$
- Linear classifier basically tries to fit a line to separate out the classes. But it fails in cases similar to multi-modal classes, which means there is a class inside of a class. e.g. cat, sitting cat, red cat, etc., 
