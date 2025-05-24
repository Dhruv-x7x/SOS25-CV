Details about this assignment can be found [on the course webpage](http://cs231n.github.io/), under Assignment #1 of Spring 2017.

## Assignment 1

### KNN 

```./cs231n/knn.ipynb``` contains an implementation of KNN on the CIFAR10 dataset. The training data included 50000 images and the testing data had 10000. I used a subset of this for the exercise, 5000 training and 500 testing. The highest accuracy for this subset was 28.8% with k = 10. I found the optimal k using k-fold cross-validation. 

On using the full dataset, I recorded the highest accuracy to be 

---

#### Numpy functions

- ```python np.array_split(array, no_of_splits)```: using this function you can easily split your training data into the number of folds that you want.
- ```python np.vstack()```: vertically stack arrays to form a matrix

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
- ```python np.hstack()```: horizontally stacks arrays to form a 1D array

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
