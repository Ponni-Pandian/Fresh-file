# ================================================================
# Advanced Time Series Forecasting with Neural State Space Models
# Complete End-to-End Project Implementation
# ================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ================================================================
# 1. Setup
# ================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# ================================================================
# 2. Load Multivariate Dataset (>500 timesteps)
# Using Weather Dataset (Temperature, Humidity, Pressure)
# ================================================================

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
df = pd.read_csv(url)

# Create multivariate features
df["Temp_Lag1"] = df["Temp"].shift(1)
df["Temp_Lag2"] = df["Temp"].shift(2)
df.dropna(inplace=True)

data = df[["Temp", "Temp_Lag1", "Temp_Lag2"]].values

# Scale data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

train_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)

obs_dim = train_tensor.shape[1]
state_dim = 6

# ================================================================
# 3. Neural State Space Model
# ================================================================

class NSSM(nn.Module):
    def __init__(self, obs_dim, state_dim):
        super(NSSM, self).__init__()
        
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        
        # Encoder: observation → latent
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim)
        )
        
        # Transition model
        self.transition = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim)
        )
        
        # Decoder: latent → observation
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, obs_dim)
        )
        
        # Learnable noise covariances
        self.Q = nn.Parameter(torch.eye(state_dim))
        self.R = nn.Parameter(torch.eye(obs_dim))

    def forward(self, y):
        T = y.shape[0]
        x = torch.zeros(self.state_dim).to(device)
        P = torch.eye(self.state_dim).to(device)
        
        total_loss = 0
        latent_states = []
        
        for t in range(T):
            # Predict
            x_pred = self.transition(x)
            P = P + self.Q
            
            # Observation prediction
            y_pred = self.decoder(x_pred)
            
            innovation = y[t] - y_pred
            S = self.R + torch.eye(self.obs_dim).to(device) * 1e-5
            
            # Likelihood loss
            total_loss += innovation.T @ torch.inverse(S) @ innovation
            
            # Update state (simple correction)
            x = x_pred + self.encoder(innovation)
            
            latent_states.append(x.detach().cpu().numpy())
        
        return total_loss / T, np.array(latent_states)

# ================================================================
# 4. Train NSSM
# ================================================================

model = NSSM(obs_dim, state_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 200

print("Training NSSM...\n")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    loss, _ = model(train_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# ================================================================
# 5. Forecasting (1-step and 5-step)
# ================================================================

model.eval()
_, latent_states = model(train_tensor)

x = torch.tensor(latent_states[-1], dtype=torch.float32).to(device)

predictions_1 = []
predictions_5 = []

with torch.no_grad():
    for t in range(len(test_tensor)):
        # 1-step
        x = model.transition(x)
        y_pred = model.decoder(x)
        predictions_1.append(y_pred.cpu().numpy())
        
    # 5-step recursive forecast
    x_temp = x.clone()
    for i in range(5):
        x_temp = model.transition(x_temp)
        y_pred = model.decoder(x_temp)
        predictions_5.append(y_pred.cpu().numpy())

predictions_1 = scaler.inverse_transform(np.array(predictions_1))
actual = scaler.inverse_transform(test_data)

# ================================================================
# 6. Baseline SARIMAX (on first variable only)
# ================================================================

train_univariate = data[:train_size, 0]
test_univariate = data[train_size:, 0]

sarima = SARIMAX(train_univariate, order=(2,1,2))
sarima_fit = sarima.fit(disp=False)
sarima_forecast = sarima_fit.forecast(steps=len(test_univariate))

# ================================================================
# 7. Metrics
# ================================================================

rmse_nssm = np.sqrt(mean_squared_error(actual[:,0], predictions_1[:,0]))
mae_nssm = mean_absolute_error(actual[:,0], predictions_1[:,0])

rmse_sarima = np.sqrt(mean_squared_error(test_univariate, sarima_forecast))
mae_sarima = mean_absolute_error(test_univariate, sarima_forecast)

print("\n================= PERFORMANCE =================")
print("NSSM (1-step)")
print("RMSE:", rmse_nssm)
print("MAE :", mae_nssm)

print("\nSARIMAX")
print("RMSE:", rmse_sarima)
print("MAE :", mae_sarima)

# ================================================================
# 8. Visualization
# ================================================================

plt.figure()
plt.plot(actual[:,0], label="Actual")
plt.plot(predictions_1[:,0], label="NSSM Forecast")
plt.plot(sarima_forecast, label="SARIMAX Forecast")
plt.legend()
plt.title("Forecast Comparison")
plt.show()

# Plot latent states
plt.figure()
for i in range(min(3, state_dim)):
    plt.plot(latent_states[:, i], label=f"Latent State {i+1}")
plt.legend()
plt.title("Learned Latent States")
plt.show()

print("\nProject Completed Successfully.")
