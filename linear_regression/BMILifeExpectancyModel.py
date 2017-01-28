import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# read data
bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')
x_values = bmi_life_data[['Life expectancy']]
y_values = bmi_life_data[['BMI']]

# train model on data
bmi_life_model = LinearRegression()
bmi_life_model.fit(x_values, y_values)

# visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, bmi_life_model.predict(x_values))
plt.show()

# Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931)
