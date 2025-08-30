from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
# load tarin and test dataset
train1 = pd.read_csv("train.csv")
test1 = pd.read_csv("test.csv")
train1["Bathroom"]= train1["BsmtFullBath"] + train1["FullBath"] + 0.5 * (train1["BsmtHalfBath"] + train1["HalfBath"]) #Create a new column "Bathroom" by combining all bathroom columns
test1["Bathroom"]= test1["BsmtFullBath"] + test1["FullBath"] + 0.5 * (test1["BsmtHalfBath"] + test1["HalfBath"]) # create a new column "Bathroom" by combining all bathroom columns
x_train = train1[["LotArea","BedroomAbvGr","Bathroom"]]
y_train = train1[["SalePrice"]]
x_test = test1[["LotArea","BedroomAbvGr","Bathroom"]]
x_train = x_train.fillna(x_train.mean())
x_test = x_test.fillna(x_test.mean())
# y_test = test1[["SalePrice"]]
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Predicted price of house: ", y_pred)
output = test1.copy()
output["PredictedPrice"] = y_pred
y_test = output.to_csv("predicted_prices.csv", index=False)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))