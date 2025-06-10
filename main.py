from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


class PlantWateringExpertSystem:
    """
    PlantWateringExpertSystem: which takes a moisture level of a soil (in %)
    then based on the following rule it will predict whether to water the plant or not.
    """
    def __init__(self, moisture_threshold=55):
        """
        Initializes the expert system.

        Args:
            learned_moisture_threshold (int): A threshold representing a value
                                             potentially learned from data
                                             (simulating a 'neuro' part).
        """
        self.threshold = moisture_threshold
        # moisture_level = moisture_readings

    def predict_watering(self, moisture_level):
        """
        Predicts whether to water the plant based on moisture level
        using symbolic rules and a learned threshold.

        Args:
            moisture_level (int): The current moisture level (e.g., from a sensor).

        Returns:
            tuple: A tuple containing the prediction ('Water' or 'No Water')
                   and a confidence score (simulated).
        """
        moisture_level = float(moisture_level)
        if moisture_level < self.threshold:
            prediction = "Soil dry, needs Water"
            confidence = 0.7 + (self.threshold - moisture_level) / self.threshold * 0.2
            confidence = min(1.0, max(0.0, confidence))

        elif moisture_level > self.threshold:
            prediction = "No need of watering"
            confidence = 0.7 + (moisture_level - self.threshold) / (60 - self.threshold) * 0.2
            confidence = min(1.0, max(0.0, confidence))

        else:
            prediction = "Undefined"
            confidence = 0.5
        
        return prediction, confidence


data = {'last_watered': [1, 5, 10, 2, 7, 12, 3, 8, 15, 4, 6, 11],
        'soil_type': [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2],
        'soil_moisture': [85, 60, 30, 90, 65, 40, 95, 70, 45, 80, 75, 50]}

df = pd.DataFrame(data)

df = pd.get_dummies(df, columns=['soil_type'], prefix='soil_type', drop_first=True)

X = df[['last_watered', 'soil_type_1', 'soil_type_2']]
y = df['soil_moisture']

model = LinearRegression()
model.fit(X, y)

print("Linear Regression Model Trained.")
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

data_for_prediction = [pd.DataFrame({'last_watered': [6],
                         'soil_type_1': [1], # Loam
                         'soil_type_2': [0]}),
                        pd.DataFrame({'last_watered': [9],
                              'soil_type_1': [0],
                              'soil_type_2': [1]}), # clay
                        pd.DataFrame({'last_watered': [7],
                              'soil_type_1': [0], # Not Loam
                              'soil_type_2': [0]}) # sand
                              ] 


predicted_moisture_loam = model.predict(data_for_prediction[0])
predicted_moisture_clay = model.predict(data_for_prediction[1])
predicted_moisture_sand = model.predict(data_for_prediction[2])


symbolic_decision = PlantWateringExpertSystem()

prediction, confidence = symbolic_decision.predict_watering(predicted_moisture_loam[0])
print("----test for loam----")
print(f"\nPredicted soil moisture for last watered 6 hours ago and loam soil: {predicted_moisture_loam[0]:.2f}%")
print(f"Symbolic Decision: {prediction} \nConfidence: {confidence:.2f}")

prediction, confidence = symbolic_decision.predict_watering(predicted_moisture_clay[0])
print("\n----test for clay----")
print(f"Predicted soil moisture for last watered 9 hours ago and clay soil: {predicted_moisture_clay[0]:.2f}%")
print(f"Symbolic Decision: {prediction} \nConfidence: {confidence:.2f}")

prediction, confidence = symbolic_decision.predict_watering(predicted_moisture_sand[0])
print("\n----test for sand----")
print(f"Predicted soil moisture for last watered 7 hours ago and sand soil: {predicted_moisture_sand[0]:.2f}%")
print(f"Symbolic Decision: {prediction} \nConfidence: {confidence:.2f}")

