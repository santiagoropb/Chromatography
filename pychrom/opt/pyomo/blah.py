import yaml

index_data = {}
index_data['A'] = dict()
index_data['A']['eqsma_km'] = 0.0
index_data['A']['eqsma_ke'] = 0.0
index_data['A']['eqsma_nu'] = 0.0
index_data['A']['eqsma_sigma'] = 0.0

index_data['B'] = dict()
index_data['B']['eqsma_km'] = 12.0/60.0
index_data['B']['eqsma_ke'] = 0.0039
index_data['B']['eqsma_nu'] = 4.4
index_data['B']['eqsma_sigma'] = 10

index_data['C'] = dict()
index_data['C']['eqsma_km'] = 22.0/60.0
index_data['C']['eqsma_ke'] = 0.0077
index_data['C']['eqsma_nu'] = 3.7
index_data['C']['eqsma_sigma'] = 12

scalar_data = dict()
scalar_data['eqsma_lambda'] = 1200.0

data = {}
data['index parameters'] = index_data
data['scalar parameters'] = scalar_data

with open("eqsma1.yml", 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)

# data for column
index_data = {}
index_data['A'] = dict()
index_data['A']['film_diffusion'] = 0.0
index_data['A']['init_c'] = 50.0
index_data['A']['init_q'] = 1200
index_data['A']['par_surfdiffusion'] = 0.0

index_data['B'] = dict()
index_data['B']['film_diffusion'] = 0.0
index_data['B']['init_c'] = 0.26
index_data['B']['init_q'] = 0.0
index_data['B']['par_surfdiffusion'] = 0.0

index_data['C'] = dict()
index_data['C']['film_diffusion'] = 0.0
index_data['C']['init_c'] = 0.5
index_data['C']['init_q'] = 0.0
index_data['C']['par_surfdiffusion'] = 0.0


scalar_data = dict()
scalar_data['binding'] = 'bm'
scalar_data['col_dispersion'] = 5.75e-8
scalar_data['col_length'] = 0.105
scalar_data['col_porosity'] = 0.7
scalar_data['par_porosity'] = 0.75
scalar_data['par_radius'] = 9e-9
scalar_data['velocity'] = 0.08454

data = {}
data['index parameters'] = index_data
data['scalar parameters'] = scalar_data

with open("column_eq1.yml", 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)
