Problem Statement and Objectives

The problem statement for this project was to predict HDB prices in Singapore for a prospective home buyer. The primary objectives were to develop a machine learning model and evaluate its predictive performance using Root Mean Squared Error (RMSE) and R-squared (R²) metrics.

Dataset and Features

The dataset used in this project was sourced from the DSI-SG Project 2 Regression Challenge (HDB Price) on Kaggle. It contained comprehensive information on historical HDB transaction prices, property features, location attributes, and other relevant factors.

[Link to data files](https://drive.google.com/drive/folders/1nGDPMmpSGhZkG0nxu7xSuVINlZYQoqkV?usp=sharing)

Methodology

The project's methodology involved several key steps:

    Data preprocessing: The dataset was cleaned and preprocessed, including handling missing values, encoding categorical variables, and normalizing numerical features.
    Feature selection: Exploratory data analysis identified strong predictive features for resale price, while outliers were also identified. Outliers were not dropped from the dataset, as they represented legitimate observations. Instead, feature engineering and automatic feature selection were performed to ensure relevant predictors were used.
    Model selection: Various regression algorithms were evaluated, including linear regression, Lasso-regularized linear regression, and Ridge-regularized linear regression, to determine the best-performing model.
    Model training and evaluation: The selected model was trained on the dataset, and its performance was evaluated using RMSE and R² metrics on a validation set.
    Hyperparameter tuning: To optimize model performance, hyperparameters such as the regularization penalty were tuned.
    Predictions: The trained model was deployed to make predictions on new data, providing estimated HDB prices based on input features.

Conclusion

The best model is a non-regularized linear regression model using 50% of the best features chosen by automatic feature selection. This model was able to predict HDB prices with a RMSE of 39,179.61 SGD and an R2 of 0.9261. The model can be used by prospective home buyers to make predictions based on a list of flat attributes.

Recommendations

There are a number of potential future work directions for this project. For example, the model could be improved by incorporating more data. Additionally, alternative machine learning algorithms can be used to improve upon the predictions by obtaining a lower RMSE and R2. Other libraries such as SHAP and LIME can be used to provide interpretability for the model.



Reference
1. https://www.hdb.gov.sg/about-us/our-role/public-housing-a-singapore-icon
2. https://www.channelnewsasia.com/commentary/emphasis-home-ownership-hdb-lease-review-of-public-housing-2071266
3. https://www.hdb.gov.sg/about-us/history
4. https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/couples-and-families/cpf-housing-grants-for-resale-flats-families
5. https://www.cpf.gov.sg/member/cpf-overview
6. https://www.worldbank.org/en/country/singapore/overview
7. https://www.ura.gov.sg/Corporate/Planning/Long-Term-Plan-Review
8. https://cnaluxury.channelnewsasia.com/exceptional-homes/singaporean-home-buying-sentiment-h2-2021-191066
9. https://www.straitstimes.com/politics/continuity-in-policies-key-to-singapores-success-says-chan-chun-sing
10. https://dreamhomessg.co/dont-wait-its-time-to-buy-why-home-prices-in-singapore-are-unlikely-to-drop-in-2023/
11. https://www.businesstimes.com.sg/property/singapore-households-net-worth-grows-residential-asset-values-climb
12. https://www.channelnewsasia.com/singapore/hdb-resale-prices-every-singapore-town-current-property-boom-3043466
13. https://www.businesstimes.com.sg/property/proportion-delayed-bto-projects-down-90-40-hdb-clear-backlog-two-years
14. https://www.businesstimes.com.sg/lifestyle/rising-appeal-resale-hdb-flats
15. https://endowus.com/insights/planning-finances-hdb-bto
16. https://sbr.com.sg/exclusive/hdb-resale-good-investment
17. https://www.teoalida.com/singapore/hdbfloorplans/
18. https://www.redbrick.sg/blog/20-housing-types-singapore-1/
19. https://blog.carousell.com/property/hdb-flat-types-singapore/
20. https://sg.finance.yahoo.com/news/different-types-hdb-houses-call-020000642.html#:~:text=Model%20A%20Maisonettes%20are%20HDB,ft%20to%201%2C551%20sq%20ft.
21. https://www.propertyguru.com.sg/property-guides/dbss-singapore-17893
22. https://getforme.com/previous2004/previous290504_hdblaunchesthepinnancleatduxton.htm
23. https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/
24. https://www.channelnewsasia.com/commentary/hdb-resale-flats-million-dollar-1-5-listings-affordability-accessibility-3364966)
25. https://www.propertyguru.com.sg/property-guides/ccr-ocr-rcr-region-singapore-ura-map-21045
26. https://www.hdb.gov.sg/residential/buying-a-flat/finding-a-flat/types-of-flats
27. https://www.finko.com.sg/articles/housing-loan/property-cooling-measures
28. https://www.channelnewsasia.com/singapore/cooling-measures-singapore-hdb-resale-prices-towns-property-map-3499961
29. https://www.ura.gov.sg/maps/
30. https://medium.com/towards-data-science/a-better-way-to-handle-missing-values-in-your-dataset-using-iterativeimputer-9e6e84857d98#f326
31. https://statisticsbyjim.com/basics/remove-outliers/
32. https://towardsdatascience.com/a-better-way-to-handle-missing-values-in-your-dataset-using-iterativeimputer-9e6e84857d98#f326
33. https://towardsdatascience.com/how-and-why-to-standardize-your-data-996926c2c832
34. https://saturncloud.io/blog/linear-regression-implementing-feature-scaling/
35. https://towardsdatascience.com/drop-first-can-hurt-your-ols-regression-models-interpretability-4ca529cfb707
36. https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html#sklearn.feature_selection.mutual_info_regression
37. https://guhanesvar.medium.com/feature-selection-based-on-mutual-information-gain-for-classification-and-regression-d0f86ea5262a
38. https://machinelearningmastery.com/information-gain-and-mutual-information/
39. https://towardsdatascience.com/understanding-entropy-the-golden-measurement-of-machine-learning-4ea97c663dc3
40. https://towardsdatascience.com/explaining-negative-r-squared-17894ca26321
41. http://www.fairlynerdy.com/what-is-r-squared/
42. https://statisticsbyjim.com/regression/root-mean-square-error-rmse/
43. https://neptune.ai/blog/feature-selection-methods
44. https://statisticsbyjim.com/regression/ols-linear-regression-assumptions/