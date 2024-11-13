# Local means-based fuzzy k-nearest neighbor classifier with Minkowski distance and relevance-complementarity feature weighting (FWM-LMFKNN)

**Introduction:** <br/>
FWM-LMFKNN-classifier improves classification accuracy by incorporating feature weights, Minkowski distance, and class representative local mean vectors. The feature weighting process is developed based on feature relevance and complementarity. The distance calculation between instances is enhanced by utilizing feature information-based weighting and Minkowski distance, resulting in a more precise set of nearest neighbors. Furthermore, the FWM-LMFKNN classifier considers the local structure of class subsets by using local mean vectors instead of individual neighbors, which improves its classification performance. 


**Matlab functions:** <br/>
The function of the FWM-LMFKNN algorithm is `FWM_LMFKNN.m`. Here, the `feat_sel_FES_RRComcorr` is the function for relevance and complementarity based feature weight generation. 


Reference:
    [Kumbure, M.M. and Luukka, P. (2024) Local means-based fuzzy k-nearest neighbor classifier with Minkowski distance and relevance-complementarity feature weighting. *Granular Computing*](https://doi.org/10.1007/s41066-024-00496-0)<br/>
<br/>
