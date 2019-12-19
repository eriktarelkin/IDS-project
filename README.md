# IDS-project

This repository contains our machine learning project (using random forest and linear regression)

It's reccomended to run the .py files in pycharm, since the project interpreter is already configured and a part of the project.

RandomForestRegression produced better results than LinearRegression.

Unfortunately we were not able to fix overfitting with some of the methods we tried(like recursive feature elimination). Later we realized that lookahead bias was the culprit of this problem(we were giving our algorithm daily high, low prices and volume. These things are not known actually until the day ends.)

Tested on AMD stock, Microsoft stock and Apple stock (RandomForestRegression worked best with AMD stock)



Some pictures: 

![amd stock](https://github.com/eriktarelkin/IDS-project/blob/master/amdstockbigbig.png)

![aapl stock](https://github.com/eriktarelkin/IDS-project/blob/master/aaplfinalpic.png)
![msft stock](https://github.com/eriktarelkin/IDS-project/blob/master/msftfinalpic.png)
