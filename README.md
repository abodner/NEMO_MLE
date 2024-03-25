# submeso_ML
Repo for data-driven submesoscale parameterization following Bodner, Balwada & Zanna (2024, in prep). The CNNs included take 8 input features ['grad_B', 'FCOR' , 'HML', 'TAU', 'Q', 'div', 'vort', 'strain'] which correspond to the normalized buoyancy gradient magnitude, Coriolis parameter, mixed layer depth, wind stress magnitude, surface heat flux, divergence, vorticity, and strain magnitude. The NEMO implementation will include the surface values of the input features and predict the mixed layer averaged vertical buoyancy flux. 

The repo contains:
* scripts 
* submeso_ml libraries 
* trained_models (pre-trained CNNs corresponding to 5 resolutions)

run pip install -e .
