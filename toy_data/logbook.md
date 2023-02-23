## 2023/01/24
Extracting repeatable code into `experiments.py` so that I can run the HPO search on a different dataset:  electricity.



## 2023/01/23
Incorporated [Optuna](https://optuna.org/) for automated hyper-parameter tuning.  So far, dropout is by far the most important parameter and it's being pushed to the most extreme values near 1.0.  I will need to understand this more deeply, as I've previously gotten best performance with small dropout and large models.  Things to try:
- Optimize accuracy rather than logloss.  Consider reducing early stop criteria, as it seems to lead to significant overfitting.
- Separate dropout on embeddings and dropout in the feed-forward (FF) layers to understand which is tending toward large values
  - For RF + NN, dropout on the FF layers is most important parameter, and it's optimized to very small values (0.001).
- Run HPO with several HPs to identify e.g., top 3 most important parameters, and then run another round to tune just those.

## 2023/01/20
Attempting to learn embeddings for leaf nodes of tree ensembles that are fit on all the features rather than trees fit on individual features (i.e., the standard practice of building tree embeddings).  Interestingly, on the `eye-movements` dataset I cannot get a NN trained in this way (purple, brown) to out perform the base tree models (blue, orange).  Rather, embeddings learned on trees that are fit on the *individual features* (red) give much better performance.

![ROC curves](/toy_data/img/eye-movements-multi-feature.png)

It's also worth noting that if using this multi-feature setup, the trees are likely highly correlated (particularly for Random Forest), and dropout on the embedding tables makes some sense and has better performance.  This would make less sense for embedding tables that map to individual features, rather than individual trees.  

Hypothesis:  A single tree splits the inputs into 100s-1000s of segments, which are then mapped to embeddings.  For an individual feature, this is a segmentation of a 1-dimensional space, whereas for the full-feature case, it's a $d$-dimensional space.  Presumably, the number of segmentations would need to increase exponentially in $d$ to have the same representation power.

For example, consider a problem where the $x^i=0$ point changes the label, so that a $d$-dimensional input space needs a function that can map $2^d$ segments of space.  Clearly, a tree will need $2^d$ leaf nodes.  An embedding table with linear classifier, however, may be able to assign constant learned values on each side of the threshold, e.g., 
$$E^{(i)}(x^i) = 
\begin{cases} 
-1 & \text{if } x^i \leq 0 \\ 
1 & \text{if } x^i > 0 
\end{cases}$$

This would be *linear* in $d$, as it would require an embedding table with 2 entries for each of the $\{x^i : i \in [d]\}$.  Then, an MLP would need to transform and aggregate the $E^{(i)}$.  In this case, would the *width* of the MLP need to scale exponentially with the number of segments?  I.e., perhaps the projection layer can figure out the segment identity based on the $E^{(i)}$, but you need a separate neuron to represent each of the segments (unless there's additional structure to exploit).  

### Action items:
- Confirm the need for 2^d scaling of the hidden layer in NNs if the input space is parsed optimally (i.e., indices denoting whether $x^i > 0$)
- Conceive of a setting where there's additional structure between the features and output, and see if this enables NN width to scale sub-exponentially, whereas tree based methods cannot escape this need.

## 2023/01/19
The key question is: will the method consistently beat GBDTs across the benchmarks?  If not, it's much less interesting to study (although the exercise may prove insightful).  There are a few variants to test:
- Feature-specific DTs
  - For numeric features, need a mechanism for differentiating between points within a single leaf node (e.g., append a rank metric, use multiple trees, interpolate between embeddings...etc.) 
- Full-feature DTs

## 2023/01/03
### XOR, multiplicative feature interactions
Playing with toy example where $y = x_0 x_1$.  This seems to make the DTE method fail.  Best hypothesis for this:  the decision trees can learn nothing about the output with a single feature, and therefore the space segmentation is not useful.  

To see this more clearly, consider the XOR problem.  Each feature can have 0/1, which will map to some embedding.  But given the value of one of the embeddings, there are equal number of data-points with `y==0` as `y==1`, corresponding to the values of the other feature.  Considering each feature independently, there's simply no way to remap it to get information about the other feature.

**Conclusion**:  using decision trees in this way does not help recover multiplicative interactions between features.  

### Using multiple features to grow trees
Hypothesis: if I allowed both features to be used to grow trees in the above problem, a tree of appropriate depth could capture the XOR relationship.  Hence, using multiple features to grow trees, and using those tree indices as embedding indices could result in the useful modeling of multiplicative relationships.  This may also help in ignoring useless features.  