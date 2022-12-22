## 1 Background

Here, we compare various methods of modeling numeric data in a binary classification task.  The goal is to better understand where standard methods succeed and fail, and why.  This problem is studied on toy data with fully specified distributions.


### 1.1 Toy data generator
A class is constructed that allows the users to add column generators with known distribution families and parameters.  Currently, the following distribution families are supported:  normal, gamma, log-normal.  For example, the following code creates a data generator and adds a column with a normal distribution, and a column with a gamma distribution:
```python
gen = ToyDataGenerator()
gen.add_normal(loc=1, scale=1)
gen.add_gamma(k=1, theta=2)
```

With this, 1000 samples of data can be generated like: `X = gen.generate(1000)`. where the first column comes from the normal distribution, and the second from the gamma.

To construct datasets for binary classification, a separate generator is constructed for each class using class-specific distributional parameters, e.g.,
```python
c1_generator = ToyDataGenerator()
c2_generator = ToyDataGenerator()

# normal - different mean, same std
c1_generator.add_normal(1, 1)
c2_generator.add_normal(2, 1)

X_1 = c1_generator.generate(1000)
X_2 = c2_generator.generate(1000)
X = np.vstack([X_1, X_2])
y = np.concatenate([np.zeros(N_1), np.ones(N_2)])
```
This data can then be shuffled and split to create ML training and test data.

### 1.2 Methods 

Our primary concern is the ability of a neural network (NN) to learn from numeric data.  In each NN experiment, a single `torch.nn.Linear` layer is used to generate the final prediction.  What differs is the data processing step, which could be one of the following methods:

* Raw:  no processing applied.
* StandardScaler ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)):  a column has the mean subtracted and is scaled by the standard deviation.
* QuantileTransformer ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)):  a non-linear transformation that maps a column to its quantiles, and then transforms the resulting cdf into a normal distribution
* Bucketize/OHE:  the input range of the column is split into N equally-sized bins, and the values are mapped into these bins.  The column is then transformed into a one-hot encoding of this bin ID.
    * For these experiments, 10 bins are created for each column.
* Decision Tree Encoding (DTE):  a decision tree is fit to the column, which is a greedy search of the input space for split points that minimizes the entropy of the target variable in the leaf nodes.  Each value is then mapped into one of the leaf nodes, giving an integer leaf node index.  This leaf node index is then used to index an embedding table, which learns a parameter for each leaf node.
    * For these experiments, trees are growth to have a maximum of 10 leaf nodes, comparable to the Bucketize/OHE method. 

As a comparison, a Random Forest ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)) model is included as a baseline, with no feature processing, using the following parameters:  `RandomForestClassifier(n_estimators=300, max_depth=8)`.  It’s worth noting that this tree depth can result in a much finer segmentation of the input space (i.e., 2^7=128 leaf nodes, compared to 10).  This is meant to represent an upper bound on performance.

## 2 Single variable experiments

### 2.1 Normal distributions

#### 2.1.1 Different mean, same standard deviation
```python
c1_generator = ToyDataGenerator()
c2_generator = ToyDataGenerator()

c1_generator.add_normal(1, 1)
c2_generator.add_normal(2, 1)
```
![Different mean, same stdev, histogram](/toy_data/img/diff-mean-same-std-hist.png | width = 100px)
![Different mean, same stdev, ROC](/toy_data/img/diff-mean-same-std-roc.png)
Here, all methods work reasonably well.  The worst method is Bucketize/OHE.  In this case, everything within a bucket is given exactly the same score, and some buckets have many more samples than others.  This coarser treatment of the input space is reflected in the ROC curve above, which large gaps between meaningful points along the curve.

The leaf node embeddings show that the learned embeddings have a monotonic relationship with the raw value, but the input space is split into non-equal sized segments.  Specifically, we see that the tails of the distributions encompass relatively large ranges of the input space, and the bins are much narrower in regions of the input space with high class overlap.
![Different mean, same stdev, embedding](/toy_data/img/diff-mean-same-std-embed.png)

#### 2.1.2 Same mean, different standard deviation
```python
c1_generator = ToyDataGenerator()
c2_generator = ToyDataGenerator()

# normal
c1_generator.add_normal(1, 1)
c2_generator.add_normal(1, 2)
```
![Same mean, different stdev, histogram](/toy_data/img/same-mean-diff-std-hist.png)
![Same mean, different stdev, ROC](/toy_data/img/same-mean-diff-std-roc.png)
The scaling transformations do not help at all here because there is no place to draw a decision boundary that separates the class due to overlapping means (i.e., you need to segment the space to extract the heavy tales on both sides).  OHE and DTE allow for this segmentation, which a linear classifier can then handle.  

Looking at the learned leaf embeddings, we see similar embedding values for the extreme tails on both ends of the distribution, and an opposite value for the mean of the distributions where one class dominates.
![Same mean, different stdev, embedding](/toy_data/img/same-mean-diff-std-embed.png)

#### 2.1.3 Different mean, different standard deviation

```python
c1_generator.add_normal(1, 1)
c2_generator.add_normal(2, 2)
```
![Different mean, different stdev, histogram](/toy_data/img/diff-mean-diff-std-hist.png)
![Different mean, different stdev, ROC](/toy_data/img/diff-mean-diff-std-roc.png)
The scaling transformations still suffer from the double-sided distribution problem.  We see the ROC curves drop below the baseline for high FPR as a consequence.

Bucketizing works better, but still suffers due to equal allocation to low-density regions of the input space.  DTE performance approximately matches RF.


### 2.2. Log-normal distributions
```python
c1_generator = ToyDataGenerator()
c2_generator = ToyDataGenerator()

# log-normal
c1_generator.add_lognormal(1, 1)
c2_generator.add_lognormal(1, 2)
```
![Log-normal, histogram](/toy_data/img/lognormal-hist.png)
![Log-normal, ROC](/toy_data/img/lognormal-roc.png)

The QuantileTransformer is able to recover distributions that look similar to the normal case, but it suffers from the same afflictions when paired with a linear classifier:
![Log-normal, histogram with log transformation](/toy_data/img/lognormal-hist-log.png)
In this case, the Bucketize/OHE also fails.  This is because it bins the input space linearly, and therefore almost all of the data is in a single bin.  In particular, despite slicing the input into 10 bins, only 5 bins had non-zero samples from the 16000 training samples, and their populations were: `[15984, 8, 6, 1, 1]`. 

When comparing the leaf node embeddings to raw features, we see similar behavior to the case of normal distributions:  extreme values are given similar embeddings.  This is hard to visualize in the raw feature domain (left), but applying a LOG transformation to the raw feature (right) shows very similar embedding structure to the normal case.
![Log-normal, embedding](/toy_data/img/lognormal-embed.png)
![Log-normal, embedding with log transformation](/toy_data/img/lognormal-embed-log.png)

### 2.3 Gamma distributions

#### 2.3.1 Single-sided
```python
c1_generator = ToyDataGenerator()
c2_generator = ToyDataGenerator()

# gamma
c1_generator.add_gamma(1, 1)
c2_generator.add_gamma(1, 2)
```
![Gamma, single-sided, histogram](/toy_data/img/gamma-hist.png)
![Gamma, single-sided, ROC](/toy_data/img/gamma-roc.png)
In this case, the normal scaling methods work well because the separation of these distributions is on a single tail and therefore a sensible decision boundary can be drawn at a single value.  

The one method that doesn’t work well is Bucketize/OHE.  This is likely due, once again, to the linearly segmentation of the heavy-tailed input space that results in some buckets having many more samples than others.  In this particular case, 9 of a possible 10 buckets had non-zero items, and the allocation of 16,000 samples is: `[12190, 2692, 752, 240, 76, 38, 8, 3, 1]`.


#### 2.3.2 Double-sided
```python
c1_generator = ToyDataGenerator()
c2_generator = ToyDataGenerator()

c1_generator.add_gamma(5, 3)
c2_generator.add_gamma(2, 8)
```
![Gamma, double-sided, histogram](/toy_data/img/gamma-double-hist.png)
![Gamma, double-sided, ROC](/toy_data/img/gamma-double-roc.png)
The raw score and scaling methods have the same problem with predictive value lying on both sides of the mean.  This phenomena seems to show up as an S-shaped ROC curve that is above the random baseline at low FPR, and then dips below for higher FPRs.  

Due to the heavy tail, we also see the Bucketize/OHE strategy struggle.  The leaf node embeddings show a double-sided mapping of the distributions, similar to the case of normal distributions with the same mean, but different variances.

![Gamma, double-sided, embedding](/toy_data/img/gamma-double-embed.png)


## 3 Summary of findings

1. Scaling and segmentation are two different problems.  Scaling will not help you separate overlapping two-sided distributions.
2. Bucketize/OHE and DTE both segment the space, but DTE is better because it can find more-optimal ways to segment the space to handle problems like, e.g., most of the buckets not having any samples
