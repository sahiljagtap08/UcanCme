# UcanCme
### UcanCme: Swimmer Position Estimation Using RNN-LSTM Based on Sensor Data (NO GPS)

#### Project Overview

In recent years, advancements in wearable technology and artificial intelligence have opened up new avenues for enhancing safety and performance in various domains. This project focuses on using Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) units to estimate the position of a swimmer in a water body based on sensor data. The primary goal is to develop a robust model that can accurately predict the swimmer's location, even in scenarios where GPS signals may be unavailable or unreliable, such as underwater environments.

#### Motivation

The motivation behind this project stems from the need to improve the safety and tracking of swimmers, divers, and marine professionals. Traditional GPS-based systems are ineffective underwater due to signal attenuation. Hence, there is a significant need for an alternative solution that can leverage available sensor data to estimate positions accurately. This project aligns with the "AI for Good" initiative, aiming to harness artificial intelligence to address real-world challenges and improve human safety and well-being.

#### Tools and Technologies Used

- **Python**: The primary programming language used for data processing, model development, and evaluation.
- **Pandas**: Used for data manipulation and analysis.
- **NumPy**: Utilized for numerical operations and array manipulations.
- **Scikit-learn**: Employed for preprocessing tasks such as data normalization and splitting.
- **TensorFlow and Keras**: Used to build, train, and evaluate the LSTM model.
- **Matplotlib**: Utilized for data visualization and plotting results.
- **Intel® Tiber™ Developer Cloud**

#### Data Collection

We collected real sensor data by actually going to a nearby swimming pool near Berkeley and recording data while swimming. This data includes:
- Rotation rates (rotationRateX, rotationRateY, rotationRateZ)
- Gravity components (gravityX, gravityY, gravityZ)
- Acceleration (accelerationX, accelerationY, accelerationZ)
- Orientation (quaternionW, quaternionX, quaternionY, quaternionZ)
- Time and seconds elapsed

The data was captured at a frequency of 100 Hz during a 100-yard swim in a 50-yard pool, covering two laps.

#### Approach

1. **Data Preprocessing**:
   - Loaded and inspected the data to ensure quality and consistency.
   - Normalized the sensor data to bring all features to a comparable scale.
   - Created sequences of data for LSTM input, capturing the temporal dependencies in the sensor readings.

2. **Model Development**:
   - Built an LSTM model with two LSTM layers followed by dropout layers to prevent overfitting.
   - Compiled the model using the mean squared error (MSE) loss function and the Adam optimizer.
   - Trained the model on the preprocessed data, ensuring a proper split between training and validation sets.

3. **Hyperparameter Tuning**:
   - Experimented with different hyperparameters such as batch size, number of epochs, learning rate, and the number of LSTM units.
   - Used grid search to find the optimal combination of hyperparameters that yielded the best performance.

4. **Model Evaluation**:
   - Evaluated the model on the validation set using MSE to assess its accuracy.
   - Visualized the predicted positions against the actual positions to understand the model's performance.

5. **Real-time Demonstration**:
   - Developed a method to load new sensor data, preprocess it, and use the trained model to predict positions in real-time.
   - Implemented a visualization tool to display the swimmer's estimated positions, making it easier to understand and interpret the results.

#### Challenges Faced

One of the main challenges we faced was the collection of sufficient training data. Given the limited time frame of the 24-hour hackathon and the fact that we were only two people, it was difficult to gather a large amount of data. We went to a nearby swimming pool to collect real sensor data while swimming, which provided us with a practical and realistic dataset. However, to improve the model's performance and accuracy, more data is needed. This challenge is fixable, and future work on the project will focus on collecting more extensive datasets.

Another challenge was ensuring the model's accuracy and reducing the difference between actual and predicted values. This required extensive hyperparameter tuning and experimentation with different model architectures.

#### Applications and Impact

The UcanCme project has significant potential to enhance safety and tracking for various individuals:
- **Marine Commandos**: In scenarios where GPS signals are jammed or unavailable, the model can provide reliable position estimates based on sensor data.
- **Recreational Swimmers and Divers**: Ensuring the safety of individuals in water bodies by providing accurate position tracking.
- **Emergency Situations**: Assisting in search and rescue operations where GPS may not be effective.

#### What's Next for UcanCme

The next steps for UcanCme include:
- **Data Collection**: Gathering more extensive datasets to improve model accuracy and robustness.
- **Model Refinement**: Experimenting with advanced model architectures and techniques to enhance performance.
- **Real-world Testing**: Implementing and testing the model in real-world scenarios to validate its effectiveness.
- **User Interface Development**: Creating a user-friendly interface to display real-time position estimates and make the tool accessible to a broader audience.
- **Collaboration**: Partnering with organizations and professionals in the swimming and marine industries to refine and deploy the solution.

By addressing these next steps, UcanCme aims to become a reliable tool for enhancing safety and tracking in water environments, contributing to the broader goal of using AI for the greater good.
