# Nicholas Califano
### ML/AI Portfolio

I've written some examples projects highlighting a variety of machine learning techniques. The projects are written to be a discussion of the techniques rather than an optimized application. 

The performance (accuracy) of these methods is often acceptable but not spectacular due to intentionally using small datasets and small models. The goal of this portfolio is to show competency with these techniques, not state-of-the-art results. 

### Examples
| Example          | Dataset                                     | Demonstration of:                                   | Results                        | Potential Future Work               |
|------------------|---------------------------------------------|-----------------------------------------------------|--------------------------------|--------------------------------|
| house_prices.ipynb | [Housing Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) | Relevant Libraries - Sklearn, Pandas <br /><br /> Model - Gradient Boosted Regression Trees <br /><br /> ML Techniques - Regression Trees, Ensemble Methods, Bagging, Boosting, Hyperparameter Search, Crossfold Validation <br /><br /> Other - Regression, Data Cleaning, Supervised Learning | ~0.90 R2 on Validation Data | - Feature Engineering <br /> - Better Data Cleaning | 
| cifar.ipynb        | [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) | Relevant Libraries - Tensorflow, Keras <br /><br /> Models - Convolutional Neural Networks, Pretrained ResNet18 <br /><br /> ML Techniques - Data Augmentation, Regularization, Transfer Learning <br /><br />  Other - Image Classification, Examples of Discussions with Stakeholders | Up to ~85% Accuracy, 10 Categories, on Test Data | - Pretrain Custom Model on a Similar Dataset (CIFAR100 minus CIFAR10 classes) <br /> - Experiment with more Data Augmentation <br /> - Experiment with better Learning Rate Scheduling  |