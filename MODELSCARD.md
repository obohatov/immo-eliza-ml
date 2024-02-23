# Model card

## Project context
This project aims to build a model to predict property prices in Belgium. The model will be used to help buyers and sellers make informed decisions about the real estate market.

## Data
The dataset consists of 75,508 properties in Belgium with the following features:
<ul>
    <li><strong>id:</strong> Property ID (int)</li>
    <li><strong>price:</strong> Property price (float)</li>
    <li><strong>property_type:</strong> Property type (e.g., apartment, house, land) (string)</li>
    <li><strong>subproperty_type:</strong> Subproperty type (e.g., flat, studio, duplex) (string)</li>
    <li><strong>region:</strong> Region (e.g., Brussels, Flanders, Wallonia) (string)</li>
    <li><strong>province:</strong> Province (string)</li>
    <li><strong>locality:</strong> Locality (string)</li>
    <li><strong>zip_code:</strong> Zip code (string)</li>
    <li><strong>latitude:</strong> Latitude (coordinates) (float)</li>
    <li><strong>longitude:</strong> Longitude (coordinates) (float)</li>
    <li><strong>construction_year:</strong> Construction year (int)</li>
    <li><strong>total_area_sqm:</strong> Total area in square meters (float)</li>
    <li><strong>surface_land_sqm:</strong> Surface land in square meters (float)</li>
    <li><strong>nbr_frontages:</strong> Number of frontages (int)</li>
    <li><strong>nbr_bedrooms:</strong> Number of bedrooms (int)</li>
    <li><strong>equipped_kitchen:</strong> Equipped kitchen (bool)</li>
    <li><strong>terrace_sqm:</strong> Terrace area in square meters (float)</li>
    <li><strong>garden_sqm:</strong> Garden area in square meters (float)</li>
    <li><strong>state_building:</strong> State of the building (e.g., good, average, bad) (string)</li>
    <li><strong>primary_energy_consumption_sqm:</strong> Primary energy consumption per square meter (float)</li>
    <li><strong>epc:</strong> Energy performance certificate (string)</li>
    <li><strong>heating_type:</strong> Heating type (e.g., gas, oil, electricity) (string)</li>
    <li><strong>cadastral_income:</strong> Cadastral income (float)</li>
    <li><strong>fl_furnished:</strong> Furnished (bool)</li>
    <li><strong>fl_open_fire:</strong> Open fire (bool)</li>
    <li><strong>fl_terrace:</strong> Terrace (bool)</li>
    <li><strong>fl_garden:</strong> Garden (bool)</li>
    <li><strong>fl_swimming_pool:</strong> Swimming pool (bool)</li>
    <li><strong>fl_floodzone:</strong> Flood zone (bool)</li>
    <li><strong>fl_double_glazing:</strong> Double glazing (bool)</li>
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