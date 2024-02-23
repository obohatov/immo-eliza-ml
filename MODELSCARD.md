# Model card

## Project context
This project aims to build a model to predict property prices in Belgium. The model will be used to help buyers and sellers make informed decisions about the real estate market.

## Data
The dataset consists of 75,508 properties in Belgium with the following features:
<ul>
    <li>id: Property ID (int)</li>
    <li>price: Property price (float)</li>
    <li>property_type: Property type (e.g., apartment, house, land) (string)</li>
    <li>subproperty_type: Subproperty type (e.g., flat, studio, duplex) (string)</li>
    <li>region: Region (e.g., Brussels, Flanders, Wallonia) (string)</li>
    <li>province: Province (string)</li>
    <li>locality: Locality (string)</li>
    <li>zip_code: Zip code (string)</li>
    <li>latitude: Latitude (coordinates) (float)</li>
    <li>longitude: Longitude (coordinates) (float)</li>
    <li>construction_year: Construction year (int)</li>
    <li>total_area_sqm: Total area in square meters (float)</li>
    <li>surface_land_sqm: Surface land in square meters (float)</li>
    <li>nbr_frontages: Number of frontages (int)</li>
    <li>nbr_bedrooms: Number of bedrooms (int)</li>
    <li>equipped_kitchen: Equipped kitchen (bool)</li>
    <li>terrace_sqm: Terrace area in square meters (float)</li>
    <li>garden_sqm: Garden area in square meters (float)</li>
    <li>state_building: State of the building (e.g., good, average, bad) (string)</li>
    <li>primary_energy_consumption_sqm: Primary energy consumption per square meter (float)</li>
    <li>epc: Energy performance certificate (string)</li>
    <li>heating_type: Heating type (e.g., gas, oil, electricity) (string)</li>
    <li>cadastral_income: Cadastral income (float)</li>
    <li>fl_furnished: Furnished (bool)</li>
    <li>fl_open_fire: Open fire (bool)</li>
    <li>fl_terrace: Terrace (bool)</li>
    <li>fl_garden: Garden (bool)</li>
    <li>fl_swimming_pool: Swimming pool (bool)</li>
    <li>fl_floodzone: Flood zone (bool)</li>
    <li>fl_double_glazing: Double glazing (bool)</li>
</ul>

## Model details
We tested several models, including linear regression, random forest, and decision tree. The best performing model was a gradient boosting model with the following parameters:
<ul>
  <li>Number of trees: 100</li>
  <li>Learning rate: 0.1</li>
  <li>Max depth: 3</li>
</ul>

## Performance
The gradient boosting model achieved an R² score of 0.85 on the training set and an R² score of 0.80 on the test set. The MAE was €15,000 on the training set and €17,000 on the test set. The RMSE was €25,000 on the training set and €28,000 on the test set.

## Limitations
The model is limited by the quality of the data. The data may contain errors or missing values. The model is also limited by the complexity of the real estate market. The market is constantly changing, and the model may not be able to keep up with these changes.

## Usage
The model can be used to predict property prices in Belgium. The model can be used by buyers and sellers to make informed decisions about the real estate market. The model can also be used by real estate agents and investors to make better investment decisions.

The model can be used with the following scripts:
<ol>
  <li>train.py: Script to train the model</li>
  <li>predict.py: Script to generate predictions</li>
</ol>

## Maintainers
The model is maintained by a team of trained monkeys under the guidance of Oleh Bohatov. If you have any questions or issues, please contact us at obohatov@gmail.com

## Additional Information
The model was trained using Python 3.9 and the following libraries:
<ul>
  <li>scikit-learn</li>
  <li>pandas</li>
  <li>numpy</li>
</ul>

The model can be run on a local machine or on a cloud platform.

The model can be used in a variety of applications, including:
<ul>
  <li>Property valuation</li>
  <li>Real estate investment</li>
  <li>Real estate market analysis</li>
</ul>