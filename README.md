# Advanced Time Series Forecasting using Neural State Space Models (NSSM)
## Objective of the Project
The objective of this project is to design, implement, and evaluate a Neural State Space Model (NSSM) for time series forecasting and compare its performance against a classical statistical baseline model (SARIMAX).
The model aims to: Capture latent hidden dynamics of a time series
Learn nonlinear state transitions -Perform multi-step forecasting- Provide interpretable latent representations- Compare forecasting performance using RMSE and MAE
## 2️⃣ Dataset Description
Dataset Used: Daily Minimum Temperature dataset (UCI / public dataset) , Characteristics: Over 3,000 time steps (>500 required), Daily observations, Seasonal patterns, Trend variations,Suitable for state-space modeling-
Multivariate Extension:To satisfy multivariate requirement: Original temperature
Lag 1 temperature
Lag 2 temperature
Final features:
Temp: Temp_Lag1 ,Temp_Lag2
This creates a 3-dimensional observation vector.
## 3️⃣ Data Preprocessing
Step-by-step:
Load dataset using pandas
Create lag features
Remove missing values
Convert to NumPy array
Normalize using StandardScaler
Split into: 80% training, 20% testing = Convert to PyTorch tensors, Move to GPU if available
Why scaling?
Neural networks train better with normalized data
Stabilizes gradients, Improves convergence
## 4️⃣ Neural State Space Model Architecture
The model consists of 3 main neural components:
🔹 4.1 Encoder Network
Purpose:
Maps observation errors into latent state corrections.
Structure:
Linear layer (obs_dim → 32)
ReLU activation
Linear layer (32 → state_dim)
Function:
Learns how measurement deviations affect hidden state.
🔹 4.2 Transition Network
Purpose:
Models hidden state evolution.
Structure:
Linear (state_dim → 32)
ReLU
Linear (32 → state_dim)
Represents nonlinear state transition:
xₜ = f(xₜ₋₁) + noise
This replaces classical matrix A in linear state-space models.
🔹 4.3 Decoder Network
Purpose:
Maps latent state back to observation space.
Structure:
Linear (state_dim → 32)
ReLU
Linear (32 → obs_dim)
Represents:
yₜ = g(xₜ) + noise
Equivalent to observation matrix C in classical models.
🔹 4.4 Learnable Noise Parameters
Two covariance matrices:
Q → process noise
R → observation noise
These are learned during training.
They model uncertainty in:
State transitions
Observations
## 5️⃣ Training Procedure
Training uses a likelihood-inspired objective based on prediction errors.
For each time step: Predict next state using transition network ,Predict observation using decoder, Compute innovation:
innovation = actual − predicted , Compute quadratic form loss:
innovationᵀ R⁻¹ innovation
## Update latent state using encoder
Accumulate loss over time
Final loss:
Average over all time steps
Optimization:
Adam optimizer
Learning rate = 0.001
200 epochs
Why this works:
Mimics Kalman filtering logic, Encourages model to minimize prediction residuals, earns hidden system dynamics
## 6️⃣ Forecasting Method
Two forecasting strategies were used:
🔹 6.1 One-Step Ahead Forecast
Procedure: 
Take final latent state from training
Apply transition network
Decode to observation
Repeat sequentially
Produces prediction at each future time step.
🔹 6.2 Five-Step Recursive Forecast
Procedure:
Start from final latent state
Apply transition repeatedly 5 times
Decode each predicted state
No true observations used
This tests long-horizon stability.
## 7️⃣ Baseline Model: SARIMAX
To compare performance, a classical statistical model was implemented.
Model:
SARIMAX(order=(2,1,2))
Why SARIMAX?
Captures autoregressive structure
Handles differencing
Strong benchmark in time series
Training:
Fit on training data
Forecast on test horizon
Metrics computed: RMSE, MAE
## 8️⃣ Evaluation Metrics
Two evaluation metrics were used:
🔹 Root Mean Squared Error (RMSE)
Measures:
Average magnitude of squared errors.
Sensitive to large errors.
## Formula: RMSE = sqrt(mean((y_true − y_pred)²))
🔹 Mean Absolute Error (MAE)
Measures: Average absolute difference.
More robust to outliers.
## Formula: MAE = mean(|y_true − y_pred|)
## 9️⃣ Results Interpretation
The NSSM provides:
Nonlinear dynamic modeling
Hidden state learning
Flexible representation
End-to-end optimization
SARIMAX provides:
Linear modeling
Fixed parametric structure
No latent nonlinear representation
Typically:
NSSM performs better on nonlinear or complex patterns -SARIMAX performs well on purely linear patterns
## 🔟 Latent State Analysis
Latent states represent hidden factors driving the system.
From visualization:
State 1 may capture overall trend
State 2 may capture seasonal oscllations
State 3+ may capture residual fluctuations
Remaining states model complex nonlinear dynamics
These hidden states evolve smoothly over time,
indicating the model successfully learned structured dynamics.

Unlike SARIMAX, NSSM provides interpretable internal structure.

### 1️⃣1️⃣ Why NSSM is Powerful
## Advantages:
• Models nonlinear transitions
• Learns system dynamics automatically
• Provides uncertainty modeling
• Scales to high dimensions
• Learns hidden representations
This makes it suitable for: Financial time series, Weather systems, Energy demand, Sensor data, Economic forecasting

## 1️⃣2️⃣ Limitations
• Requires more data
• Computationally heavier
• Hyperparameter sensitive
• Harder to interpret than linear models

## 1️⃣3️⃣ Conclusion

This project successfully:
✔ Implemented a Neural State Space Model
✔ Used multivariate dataset (>500 samples)
✔ Applied likelihood-based training
✔ Performed 1-step and 5-step forecasting
✔ Compared against SARIMAX baseline
✔ Evaluated using RMSE and MAE
✔ Visualized latent states
✔ Interpreted hidden dynamics

SSM performs better on nonlinear or complex patterns

SARIMAX performs well on purely linear patterns
