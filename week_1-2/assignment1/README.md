Details about this assignment can be found [on the course webpage](http://cs231n.github.io/), under Assignment #1 of Spring 2017.

## Assignment 1

### KNN 

```./cs231n/knn.ipynb``` contains an implementation of KNN on the CIFAR10 dataset. The training data included 50000 images and the testing data had 10000. I used a subset of this for the exercise, 5000 training and 500 testing. The highest accuracy for this subset was 28.8% with k = 10. I found the optimal k using k-fold cross-validation. 

On using the full dataset, I recorded the highest accuracy to be 37% after computing for 60 minutes. I can conclude that KNN is not good for images as computation takes too long during prediction time and the accuracy is very low even for a decently large training dataset. 

---

#### Numpy functions

- ```np.array_split(array, no_of_splits)```: using this function you can easily split your training data into the number of folds that you want.
- ```np.vstack()```: vertically stack arrays to form a matrix

```python
import numpy as np
# Example 1: Stacking 1D arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 8, 9])

result = np.vstack([a, b, c])
print(result)
# Output:
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]
print(f"Shape: {result.shape}")  # Shape: (3, 3)
```
- ```np.hstack()```: horizontally stacks arrays to form a 1D array

```python
# Example 1: Stacking 1D arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 8, 9])

result = np.hstack([a, b, c])
print(result)
# Output: [1 2 3 4 5 6 7 8 9]
print(f"Shape: {result.shape}")  # Shape: (9,)
```

---

#### Finding KNN in py

```python
closest_y = self.y_train[np.argsort(dists[i])[:k]]
y_pred[i] = Counter(closest_y).most_common(1)[0][0]
```

With just two lines we can find the k closest points to our input image. `dists` array contains the distances of each y_train to x_train. Each row is y_train[i] and every column contains differences of all y_train to x_train[j]. `np.argsort` sorts in increasing order, then we slice the first k smallest values. The second line calls on `Counter()` from `collections` which has a `.most_common()` attribute and using this we can find the majority class. 
