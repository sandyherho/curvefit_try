import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import altair as alt
import os
plt.style.use('bmh')

# Load data from CSV file
data = np.loadtxt("atm_pco2_scenarios.csv", delimiter=",", skiprows=1)
Time = data[:, 0]  # First column is time (in years)
CO2_data = data[:, 1:]  # Remaining columns are CO2 scenarios in atm

# Convert CO2_data from atm to ppmv
CO2_data_ppmv = CO2_data * 2.1315e6

# Define the exponential model function with a single term
def exponential_model(t, alpha, beta, gamma, B):
    t0 = t[0]  # Initial time
    uE = 1000  # Fixed emission value for fitting (adjust if needed)
    # Calculate CO2 concentration using the model
    co2 = B + 1000 * 0.469 * (alpha + beta * uE) * np.exp(-(t - t0) / gamma)
    return co2

# Perform curve fitting for each scenario
fitted_params_exp = []
for j in range(CO2_data_ppmv.shape[1]):
    try:
        params, _ = curve_fit(exponential_model, Time, CO2_data_ppmv[:, j], maxfev=10000)
        fitted_params_exp.append(params)
    except RuntimeError:
        print(f"Warning: Optimal parameters not found for Scenario {j+1}. Skipping this scenario.")
        fitted_params_exp.append([np.nan] * 4)  # Placeholder for failed fits

# Calculate fitted CO2 values
co2_fitted_exp = np.zeros_like(CO2_data_ppmv)
for j in range(CO2_data_ppmv.shape[1]):
    if not np.isnan(fitted_params_exp[j][0]):  # Check if fit was successful
        co2_fitted_exp[:, j] = exponential_model(Time, *fitted_params_exp[j])

# Create a directory to save plots
os.makedirs("plots", exist_ok=True)  # Create if it doesn't exist

# --- Plotting ---

# Plot original data (converted to ppmv)
plt.figure(figsize=(10, 6))
for j in range(CO2_data_ppmv.shape[1]):
    plt.semilogx(Time, CO2_data_ppmv[:, j], 'o', label=f'Scenario {j+1}')
plt.xlabel('Year (log scale)')
plt.ylabel('Atmospheric CO$_2$ (ppmv)')
plt.xlim(min(Time), max(Time))
plt.ylim(0, CO2_data_ppmv.max())
plt.legend()
plt.title('Original CO2 Emission Scenarios (ppmv)')
plt.savefig("plots/original_co2_scenarios_ppmv.png", dpi=300)

# Plot fitted curves (exponential model, in ppmv)
plt.figure(figsize=(10, 6))
for j in range(CO2_data_ppmv.shape[1]):
    if not np.isnan(fitted_params_exp[j][0]):  # Check if fit was successful
        plt.semilogx(Time, co2_fitted_exp[:, j], label=f'Scenario {j+1}')
plt.xlabel('Year (log scale)')
plt.ylabel('Atmospheric CO$_2$ (ppmv)')
plt.xlim(min(Time), max(Time))
plt.ylim(0, np.nanmax(co2_fitted_exp))
plt.legend()
plt.title('Fitted CO2 Emission Scenarios (Exponential Model, ppmv)')
plt.savefig("plots/fitted_co2_scenarios_exp_ppmv.png", dpi=300)

# --- Save Fitted Equations ---

# Function to format the equation
def format_equation(params):
    alpha, beta, gamma, B = params
    uE = 1000
    t0 = Time[0]
    return f"CO2(t) = {B:.2f} + 1000 * 0.469 * ({alpha + beta * uE:.6f}) * exp(-(t - {t0:.2f}) / {gamma:.6f})"

# Print and save fitted equations
equations = []
for j, params in enumerate(fitted_params_exp):
    if not np.isnan(params[0]):  # Check if fit was successful
        equation = format_equation(params)
        print(f"\nFitted Equation for Scenario {j+1} (Exponential Model):")
        print(equation)
        equations.append(equation)

# Save equations to a text file
with open("fitted_equations_exp_ppmv.txt", "w") as f:
    for equation in equations:
        f.write(equation + "\n\n")

# --- Altair Plots ---

# Create DataFrame for Altair plot (only for fitted data with exponential model)
df_fitted_exp = pd.DataFrame({'Year': Time})
for j in range(CO2_data_ppmv.shape[1]):
    if not np.isnan(fitted_params_exp[j][0]):  # Check if fit was successful
        df_fitted_exp[f'Scenario {j+1} (Fitted)'] = co2_fitted_exp[:, j]

# Melt the DataFrame to long format
df_melted_exp = df_fitted_exp.melt(id_vars='Year', var_name='Scenario', value_name='CO2')

# Create Altair chart (only for fitted data with exponential model)
chart_exp = alt.Chart(df_melted_exp).mark_line(point=True).encode(
    x=alt.X('Year:Q', scale=alt.Scale(type='log')),
    y='CO2:Q',
    color='Scenario:N',
    tooltip=['Year', 'Scenario', 'CO2']
).properties(
    title='Curve Fitting of CO2 Emission Scenarios (Exponential Model, ppmv, Altair)'
).interactive()

# Save the chart
chart_exp.save('co2_emissions_scenarios_curve_fitting_fitted_exp_ppmv.json')

