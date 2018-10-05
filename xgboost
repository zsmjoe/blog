
## A Simple Exploration about XGBoost.

The XGboost ( eXtreme Gradient Boosting ) is an algorithm that has recently been dominating applied machine learning and Kaggle competitions. The very first time that I saw it is when I participated in the competition of [Houseprice](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) in Kaggle and recently I am doing the stage in the MediaTech Solutions (SaaS) in paris. There is a project that I employed this algorithm again, so in order to master this algorithm completely I decide to write this blog.

There are two main parts in this blog. First, I try to demonstrate how the models works including its prediction function, objective function, exact greedy algorithm.

## Prediction model

Generally, for a tree model, we could assume its prediction model like this:

![tree-prediction model](https://cdn-images-1.medium.com/max/2000/1*Wy75mqTedPJpO8zsycZ6Ag.gif)

![](https://cdn-images-1.medium.com/max/2000/1*SgnfFam1fmUuTv-xdlPspg.gif)

and specifically F is the space of the regression tree (known as CART) Here q represents the structure of each tree that maps an example to the corresponding leaf index. T is the number of leaves in the tree. Each f(k) corresponds to an independent tree structure q and leaf weights w.

The theory can be vividly explained by the following graph.

![regression tree models](https://cdn-images-1.medium.com/max/2624/1*PUVeE7W05MkKGdxK_V2W3A.png)

## Objective function

The tree ensemble model that we showed above includes functions as parameters and cannot be optimized using traditional optimization methods in Euclidean space. Instead, the model is trained in an additive manner. The objective function is shown as following:

![Objective function](https://cdn-images-1.medium.com/max/2000/1*ohIy7AN4or8PvWChVN7WWg.gif)

to be more specific,

![is the prediction of the i-th instance at the t-th iteration](https://cdn-images-1.medium.com/max/2000/1*QGG2aa4ApdVYa4uWtcqCvg.gif)

It is a diﬀerentiable convex loss function that measures the diﬀerence between the prediction and the target. The second term omega penalizes the complexity of the model (i.e., the regression tree functions). The additional regularization term helps to smooth the ﬁnal learnt weights to avoid over-ﬁtting. Intuitively, the regularized objective will tend to select a model employing simple and predictive functions.There is a similiar example from this article.[1]

![regularization function](https://cdn-images-1.medium.com/max/2000/1*fB20_ecQHk8jLkNHH1bKxQ.png)

When the regularization parameter is set to zero, the objective falls back to the traditional gradient tree boosting(reference).

and according to the Taylor formula

![Taylor formula](https://cdn-images-1.medium.com/max/2000/1*bxM7XdfpPD2XBM97HMvG8A.gif)

we could make the following transformation:

![](https://cdn-images-1.medium.com/max/2000/1*kLawK5Y_p2v54FJh-bozjA.png)

where we suppose

![](https://cdn-images-1.medium.com/max/2000/1*D_j7PvmfqtaE9F5K7eQasQ.png)

are the first and second order gradient statistics on the loss function.

After we remove the constant terms, we could obtain the following simplified objective at step t:

![simplified objective function](https://cdn-images-1.medium.com/max/2000/1*_3904hijl4e_Wck4GTUbtQ.png)

Someone may be confusing why I am trying to demonstrate the equations above. Although it is a little complicated, but we could be more clear about our ultimate goal and the part that we really need to devote to.Such a form contains **all objective functions that can be derived**, in other words, with this form, the code we write can be used to solve **various problems including regression, classification and sorting**.

Next step is quite essential. According to [2] Tianqi Chen in his article ***XGBoost: A Scalable Tree Boosting System , ***we could define

![](https://cdn-images-1.medium.com/max/2000/1*hEElhqWBntNk6_smYgDdBQ.png)

specify q(x)

and we could have

![](https://cdn-images-1.medium.com/max/2000/1*m8ClyN5M6ZqfyKSyM8vi4g.png)

This function consists of T independent quadratic functions. Each has only one independent variable. We can define

![](https://cdn-images-1.medium.com/max/2000/1*VQamLfEcjDpI039ElujRUA.png)

Thus we could have

![](https://cdn-images-1.medium.com/max/2000/1*bcvimcbPsbJ06nfwz-O4uA.png)

where function Obj is our final objective function. It is clear that this is a problem of the minimum of the quadratic function, so we have

![](https://cdn-images-1.medium.com/max/2000/1*Etx6d5aoGY-W9bBPbDQQkA.png)

In fact, the final form of Obj is quite similar with that of Gini impurity which serves as an evaluation manner to judge a tree model. Here is a example from Tianqi Chen.

![](https://cdn-images-1.medium.com/max/2000/1*P0yJ62ALMKc2TPW1Sg5LHQ.png)

![](https://cdn-images-1.medium.com/max/2000/1*VCRlKBcN2XJozwsqfGLI3A.png)

change the formule

A smaller score indicates a better structure of the tree model.

## Exact Greedy Algorithm

With the objective function, one of the key problems in tree learning is to ﬁnd the best split . In order to do so, a split ﬁnding algorithm enumerates over all the possible splits on all the features. We call this the exact greedy algorithm.

The loss function after the split is given by

![Loss function](https://cdn-images-1.medium.com/max/2000/1*XUbGbLPHq-wLxHyB90s85g.png)

There are four parts in this equation. The first one indicates the score of the left split node, the second one indicates the score of the right split node, the third one indicates the score if we do not split and the last one indicates the cost if we do the split.

**It also demonstrates that when we make advances step by step, strategies like calculating scores and tree pruning will naturally appear, rather than being heuristic.**

**The last part is quite innovative.** Split does not necessarily make the model better because we have a penalty for introducing new leaves. Optimizing this target corresponds to the tree pruning. When the introduced segmentation gain is less than a threshold, we can cut this segmentation.

The pseudo code is shown as follows

![](https://cdn-images-1.medium.com/max/2000/1*-YYbzP1u41MGKDqu8pI9hg.png)

## The problems concerning our dataset.

The final goal is to predict the indicator. and there are two main problems that we have.

First is the missing values. Considering our source of data, which is the questionnaire, it is quite common that there are enormous incomplete answers. All this will lead to missing values in our training set. It is quite confusing because sometimes the percentage of missing values of a certain column could take up to more than 80%.

Another problems is the correlation. According to the common sense, there are several variables which have great correlation with indicator. However, for the test set, since indicator are missing, all these high-correlated variables are also missing. In other words, although all these kinds of variables have shown high importance during the process of training, they would be quite useless for the real prediction.

And we have more interest in the unsatisfied clients which is not as frequent as we imagine. With the lack of such kind of training data, how can we make appropriate training model?

## XGBoost for the missing values

XGBoost has its own way to deal with the missing values. It could be indicated as follows:

![solutions for missing values](https://cdn-images-1.medium.com/max/2000/1*1Q81RPiH3QMJb7If_OJ1EQ.png)

When a value is missing in the sparse matrix x, the instance is classiﬁed into the default direction. There are two choices of default direction in each branch. The optimal default directions are learnt from the data. The algorithm is shown above. The key improvement is to only visit the non-missing entries I_k. The presented algorithm treats the non-presence as a missing value and learns the best direction to handle missing values. The same algorithm can also be applied when the non-presence corresponds to a user speciﬁed value by limiting the enumeration only to consistent solutions.

## Feature Selection

preprocessing:

Since we have numerous categorical variables, I adopted the Labelcoding method to transform the strings into numeric values. For the tree model, I chose to do the labelencoding instead of one-hot encoding to avoid the disaster of dimensions

the definition of SHAP

After training the model, we would select the most important features. The metrics that I employ is SHAP ( [SHapley Additive exPlanation](https://arxiv.org/pdf/1705.07874.pdf)) values.

If you are interested in the definition of SHAP, here is [a brief introduction.](https://indico.cern.ch/event/736010/contributions/3035968/attachments/1667834/2674455/14.06.18.pdf)(click it)

Here are some examples from my case.

![](https://cdn-images-1.medium.com/max/2182/1*ijUD_XBeUQsnu6q1R4O-4Q.png)

The above explanation shows features each contributing to push the model output from the base value (the average model output over the training dataset we passed) to the model output. Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue.

If we take many explanations such as the one shown above, rotate them 90 degrees, and then stack them horizontally, we can see explanations for an entire dataset (in the notebook this plot is interactive):

![](https://cdn-images-1.medium.com/max/2202/1*Z4PC7u3QBzI5MS49THDOcQ.png)

To get an overview of which features are most important for a model we can plot the SHAP values of every feature for every sample. The plot below sorts features by the sum of SHAP value magnitudes over all samples, and uses SHAP values to show the distribution of the impacts each feature has on the model output. The color represents the feature value (red high, blue low). This reveals for example that a high LSTAT (% lower status of the population) lowers the predicted home price.

![](https://cdn-images-1.medium.com/max/2000/1*oe-ulzKBbfDjLPlcaQkbyg.png)

We can also just take the mean absolute value of the SHAP values for each feature to get a standard bar plot (produces stacked bars for multi-class outputs):

![](https://cdn-images-1.medium.com/max/2000/1*Hna0kIpUBQodzph4lLYeFQ.png)

Then I will do a step-wise regression, to be more concrete, it is **Backward elimination**,which means I will eliminate the feature with lowest SHAP values and redo the training process to see if the precision of test set has been improved. This loop will end until there would be no improvement.

## Reference

[1] T. Zhang and R. Johnson. Learning nonlinear functions using regularized greedy forest. IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(5), 2014.

[2] Chen, Tianqi, and Carlos Guestrin. “Xgboost: A scalable tree boosting system.” *Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining*. ACM, 2016.
