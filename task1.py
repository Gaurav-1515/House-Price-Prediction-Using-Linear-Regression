from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
train1 = pd.read_csv("train.csv")
test1 = pd.read_csv("test.csv")
train1["Bathroom"]= train1["BsmtFullBath"] + train1["FullBath"] + 0.5 * (train1["BsmtHalfBath"] + train1["HalfBath"]) #Create a new column "Bathroom" by combining all bathroom columns
test1["Bathroom"]= test1["BsmtFullBath"] + test1["FullBath"] + 0.5 * (test1["BsmtHalfBath"] + test1["HalfBath"]) # create a new column "Bathroom" by combining all bathroom columns
x_train = train1[["Id","LotArea","BedroomAbvGr","Bathroom"]]
y_train = train1[["SalePrice"]]
x_test = test1[["Id","LotArea","BedroomAbvGr","Bathroom"]]
x_train = x_train.fillna(x_train.mean())
x_test = x_test.fillna(x_test.mean())
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pred = y_pred.flatten()
print("Predicted price of house: ", y_pred)
submission = pd.DataFrame({"Id": test1["Id"], "SalePrice": y_pred})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
# Predict on training data itself (just for visualization)
y_train_pred = model.predict(x_train)
plt.scatter(y_train, y_train_pred, color="blue", alpha=0.5)
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs Predicted Prices (Train Set)")
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')  # perfect line
plt.show()
