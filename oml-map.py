import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Load and preprocess the data
def preprocess_features(data):
    data['Wind_Sin'] = np.sin(np.radians(data['Wind_Direction']))
    data['Wind_Cos'] = np.cos(np.radians(data['Wind_Direction']))
    data = data.drop(columns=['Wind_Direction'])
    data['Concentration_WindSpeed'] = data['Concentration'] * data['Wind_Speed']
    data['Wind_Speed_Squared'] = data['Wind_Speed'] ** 2
    data['Log_Concentration'] = np.log(data['Concentration'] + 1)  # Avoid log(0)
    return data

# Load the dataset
csv_file = '/Users/bogdanbora/Desktop/dust_concentration_cluj_large.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file)
data = preprocess_features(data)

# Split into features and targets
features = data[['Device_Lat', 'Device_Long', 'Wind_Speed', 'Wind_Sin', 'Wind_Cos', 
                 'Concentration', 'Concentration_WindSpeed', 'Wind_Speed_Squared', 'Log_Concentration']]
targets = data[['Source_Lat', 'Source_Long', 'Emission_Rate']]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_val_normalized = scaler.transform(X_val)

# Normalize targets for latitude and longitude
lat_scaler = StandardScaler()
long_scaler = StandardScaler()

y_train['Source_Lat'] = lat_scaler.fit_transform(y_train[['Source_Lat']])
y_val['Source_Lat'] = lat_scaler.transform(y_val[['Source_Lat']])
y_train['Source_Long'] = long_scaler.fit_transform(y_train[['Source_Long']])
y_val['Source_Long'] = long_scaler.transform(y_val[['Source_Long']])

# Build the multi-output neural network
def build_multi_output_nn(input_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    # Three outputs: Source_Lat, Source_Long, Emission_Rate
    output_lat = Dense(1, name='Source_Lat')(x)
    output_long = Dense(1, name='Source_Long')(x)
    output_emission = Dense(1, name='Emission_Rate')(x)
    model = Model(inputs=inputs, outputs=[output_lat, output_long, output_emission])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='mse', 
                  metrics={'Source_Lat': 'mae', 'Source_Long': 'mae', 'Emission_Rate': 'mae'})
    return model

# Train the multi-output neural network
nn_model = build_multi_output_nn(X_train_normalized.shape[1])
history = nn_model.fit(
    X_train_normalized,
    [y_train['Source_Lat'], y_train['Source_Long'], y_train['Emission_Rate']],
    validation_data=(
        X_val_normalized,
        [y_val['Source_Lat'], y_val['Source_Long'], y_val['Emission_Rate']]
    ),
    epochs=100,
    batch_size=32,
    verbose=1
)

# Haversine formula for distance calculation
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth's radius in meters
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Plot map with source and device
def plot_map(device_lat, device_long, predicted_lat, predicted_long):
    # Calculate the distance between the device and the source
    distance = calculate_distance(device_lat, device_long, predicted_lat, predicted_long)

    # Plot the map
    plt.figure(figsize=(8, 6))
    plt.scatter(device_long, device_lat, color='blue', s=100, label='Device Location')
    plt.scatter(predicted_long, predicted_lat, color='red', s=100, label='Predicted Source Location')
    plt.plot([device_long, predicted_long], [device_lat, predicted_lat], color='green', linestyle='--', label=f'Distance: {distance:.2f} m')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Source and Device Map')
    plt.legend()
    plt.grid(True)
    plt.savefig('source_device_map.png')
    print("Map saved as 'source_device_map.png'.")
    plt.show()

# Dispersion visualization
def visualize_dispersion(Q, u, H, device_concentration):
    x_range = np.linspace(1, 2000, 200)  # Downwind distances (m)
    y_range = np.linspace(-500, 500, 100)  # Crosswind distances (m)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i, j]
            y = Y[i, j]
            sigma_y, sigma_z = 0.1 * x, 0.05 * x  # Dispersion coefficients
            Z[i, j] = Q / (2 * np.pi * u * sigma_y * sigma_z) * np.exp(-y**2 / (2 * sigma_y**2)) * np.exp(-H**2 / (2 * sigma_z**2))

    # Scale the entire map so that the maximum value matches the input device concentration
    scaling_factor = device_concentration / Z.max()
    Z *= scaling_factor

    # Visualization
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Pollutant Concentration (scaled to µg/m³)')
    plt.title('Pollutant Dispersion (OML Multi-Model)')
    plt.xlabel('Distance Downwind (m)')
    plt.ylabel('Distance Crosswind (m)')
    plt.grid(True)

    plt.savefig('refined_dispersion_visualization.png')
    print("Dispersion visualization saved as 'refined_dispersion_visualization.png'.")
    plt.show()

# User-friendly prediction interface
def predict_source():
    try:
        # Input device parameters
        device_lat = float(input("Enter the device latitude: "))
        device_long = float(input("Enter the device longitude: "))
        wind_speed = float(input("Enter the wind speed (m/s): "))
        wind_direction = float(input("Enter the wind direction (degrees): "))
        concentration = float(input("Enter the pollutant concentration at the device (µg/m³): "))

        # Preprocess input features
        input_features = pd.DataFrame({
            'Device_Lat': [device_lat],
            'Device_Long': [device_long],
            'Wind_Speed': [wind_speed],
            'Wind_Sin': [np.sin(np.radians(wind_direction))],
            'Wind_Cos': [np.cos(np.radians(wind_direction))],
            'Concentration': [concentration],
            'Concentration_WindSpeed': [concentration * wind_speed],
            'Wind_Speed_Squared': [wind_speed ** 2],
            'Log_Concentration': [np.log(concentration + 1)]
        })
        input_features_normalized = scaler.transform(input_features)

        # Predict source location and emission rate
        predicted_lat_norm, predicted_long_norm, predicted_emission = nn_model.predict(input_features_normalized)
        predicted_lat = lat_scaler.inverse_transform(predicted_lat_norm)[0][0]
        predicted_long = long_scaler.inverse_transform(predicted_long_norm)[0][0]

        print(f"\nPredicted Source Location: Latitude {predicted_lat:.6f}, Longitude {predicted_long:.6f}")
        print(f"Predicted Emission Rate: {predicted_emission[0][0]:.2f} g/s")

        # Plot the map
        plot_map(device_lat, device_long, predicted_lat, predicted_long)

        # Visualize dispersion
        visualize_dispersion(Q=predicted_emission[0][0], u=wind_speed, H=50,  # Example: H = 50m source height
                             device_concentration=concentration)
    except ValueError as e:
        print(f"Input error: {e}")

# Run prediction interface
if __name__ == "__main__":
    predict_source()

