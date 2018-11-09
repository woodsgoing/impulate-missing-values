# impulate-missing-values
Introduction
Provide methods to impulate missing values with various of methods for nominal and numric type values.
We support tranditional missing fill method like fill with mean/mode value, or fill with stuff value just to mark.
We support algorithm based missing value impulation for more reasonable,like knn, decision tree, bayes, forest tree(ensemble tree), lgbm.
We support flexible way to impute named 'auto', which select the best imputation after try kinds of algorithm.
For reasonable imputation, we consider kinds of scenarios, like missing ratio, inexistene and missing status correlation.

Key Public APIS
process_missing() provides integrated API to handle missing values, including add_nan_ratio(), fill_nanfill_nan_status() and fill_nan('auto')
fill_nan() provides general API to impute missing values, including kinds of algorithm.

Usage
setEnvInfo() is necessary to setup log info path, before call functions to impute missing values.
Variable debug is the switcher of debug info, while trace info is always output.
Some constant values are defined as default algorithm parameter. They can be tuned if necessary, with assistant of log info.
