# Neuro-Symbolic Expert System
- This is a simple system that predicts wheather or not to water your plan based on the soil moisture levels
- The main purpose of this project is to demonstrate how neuro-symbolic ai works, a combination of neural networks and symbolic reasoning to create a rhobust system.

## Description

- The system begins training on a given dataset (I choose a small dataset just for demonstration purposes), after finishing it's training the model then predicts the moisture levels for a given soil based on two given parameters *Soil Type* and *Last watered*.
> 1. **Soil Type**: classified wheather its clay, sand or loam (0 for sand, 1 for loam, 2 for clay).
> 2. **Last Watered**: This defines the last time the soil was watered (in hours), Eg 2 hrs ago, 10 hrs ago..., This is under the assumption that the soil moisture levels were 100% when last watered.

- After getting the soil moisture prediction we feed that into the ```PlantWateringExpertSystem``` object to decide wheather or not to water the plant based on the defined rule.

# How it Works?
1. Clone the repo
2. Navigate to project dir
3. Install all the neccessary dependencies ```pip install -r requirements.txt```
4. Run the main file ```python main.py```
