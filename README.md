# codes
Repository contains codes for Spinodal, Neural network and ML methods. 
<h1>Python Scripts Documentation</h1>
The Python code for the spinodal method, multilayer feed forward neural network, and XGBoost method in the book is concisely presented in listings 1-3. The XGBoost method listing can be easily modified (basically, you just need to import other libraries) with other methods such as KNN, RF, SVR, GBDT, GB, etc.
<p>The Python scripts contained in this directory include:</p>
<ul>
  <li>Spinodal Method (listing 1)</li>
  <li>Feed Forward Neural Networks (listing 2)</li>
  <li>XGBoost Method (listing 3)</li>
  <li>GBDT (Gradient Boosting Decision Trees)</li>
  <li>GB (Gradient Boosting)</li>
  <li>RF (Random Forest)</li>
  <li>SVR (Support Vector Regression)</li>
  <li>KNN (K-Nearest Neighbors)</li>
  
  
</ul>

<hr>

<h2>Structure of the Scripts</h2>
<p>Each script is divided into two parts:</p>
<ul>
  <li><strong>Model Training</strong>: The section related to training the model.</li>
  <li><strong>Loading Pre-trained Models</strong>: The section that loads previously saved weights/metrics for further use.</li>
</ul>

<p>The scripts were developed using <em>Spyder IDE</em>, which offers similar functionality to Jupyter notebooks (i.e., the ability to run individual code blocks without executing the entire script).</p>

<hr>

<h2>Additional Features</h2>
<p>The scripts have been enhanced with grid search functionality and Bayesian optimization compared to previous versions.</p>

<hr>

<h2>Included Data</h2>
<p>The directory also contains a sample dataframe for model training and validation testing: <code>zbior_23.csv</code>.</p>

<hr>

<h2>Program Output</h2>
<p>The program generates:</p>
<ul>
  <li>The entire training and test set as a CSV file.</li>
  <li>Results of each ILS (Iterative Learning Strategy) for grid search and Bayesian methods, saved in the format:</li>
</ul>

<pre>
file_pathG = os.path.join(directory_path, name + '_U_DATA_SVR_GRID.xlsx')
file_pathB = os.path.join(directory_path, name + '_U_DATA_SVR_BAYES.xlsx')
</pre>

<p>Where <code>name</code> could be, for example, <code>C2Mim_NTF2</code>.</p>

<hr>

<h2>Important Notice!</h2>
<p>In the case of neural networks, computations take significantly longer compared to other methods. On a home test computer, the execution time exceeded <strong>47 hours</strong>.</p>

<hr>

<h2>Additional Information</h2>
<p>For the listed methods, except for XGBoost and SVR, <code>.pkl</code> files containing pre-trained models have been added. To use them, follow these steps:</p>

<h3>1. Define the test and training datasets:</h3>

<pre><code>
df = pd.read_csv('zbior_23.csv', encoding='utf8', sep=';')
X = df[['M_C', 'M_A', 'SYM', 'P', 'T']].values
y = df[['EXP U']].values

#%%
'''Data preparation - splitting into training and test sets'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

print(X_train.shape)
print(y_train.shape)
print('Test shapes')
print(X_test.shape)
print(y_test.shape)

# Feature scaling for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
</code></pre>

<h3>2. Load the predefined model:</h3>

<pre><code>
# Load the model after Grid Search
loaded_grid_model = joblib.load('best_model_grid.pkl')

# Load the model after Bayesian Optimization
loaded_bayes_model = joblib.load('best_model_bayes.pkl')

# Load the scaler
scaler = joblib.load('scaler.pkl')
</code></pre>

<hr>

<h2>How to Perform Calculations</h2>
<p>Let's assume we are working with the ionic liquid <code>C2ImC1OC8_NTF2</code>. The following steps outline how to perform the calculations:</p>

<h3>1. Define the name and parameters:</h3>

<pre><code>
nazwa = 'C2ImC1OC8_NTF2'
Mcat = 239.376  # Molecular weight of the cation
Man = 280.146  # Molecular weight of the anion
</code></pre>

<h3>2. Define the pressure and temperature arrays:</h3>
<p>The pressure values (<code>P</code>) , and the temperature values (<code>T</code>) are in Kelvin:</p>

<pre><code>
P = [
    0.10, 9.81, 19.62, 29.43, 39.24, 49.05, 
    58.86, 68.67, 78.48, 88.29, 98.10, 107.91, 
    117.72, 127.53, 137.34, 147.15
]

T = [295.15, 313.15, 333.45, 353.45, 373.15]
</code></pre>

<h3>3. Set the symmetry flag:</h3>
<p>The variable <code>issym</code> is set to <code>0</code> to indicate that the molecule does not have symmetry in this context:</p>

<pre><code>
issym = 0
</code></pre>

<h3>4. Perform the calculations:</h3>
<p>To carry out the calculations, use the following function call:</p>

<pre><code>
res_grid, res_bayes = predictions3(Mcat, Man, issym, P, T, best_model_grid, best_model_bayes, scaler)
</code></pre>

<p>This will return the grid search and Bayesian optimization results for the given input parameters.</p>

<hr>

<hr>

<h2>Flowchart of the Process</h2>

<div style="font-family: 'Courier New', Courier, monospace;">
  <pre>
    ┌───────────────────────────┐
    │  Importing Libraries      │
    └────────────┬───────────────┘
                 │
    ┌────────────▼───────────────┐
    │  Loading Data from CSV     │
    └────────────┬───────────────┘
                 │
    ┌────────────▼───────────────┐
    │  Splitting Data into Train  │
    │  and Test Sets              │
    └────────────┬───────────────┘
                 │
    ┌────────────▼───────────────┐
    │   Scaling Features          │
    └────────────┬───────────────┘
                 │
    ┌────────────▼───────────────┐
    │   Training  Model          │
    │                            │
    └────────────┬───────────────┘
                 │
    ┌────────────▼───────────────┐
    │    Predictions and Metrics  │
    │    (R² and MSE)             │
    └────────────┬───────────────┘
                 │
    ┌────────────▼───────────────┐
    │   Cross-Validation (CV)     │
    └────────────┬───────────────┘
                 │
    ┌────────────▼───────────────┐
    │   Grid Search Optimization  │
    └────────────┬───────────────┘
                 │
    ┌────────────▼───────────────┐
    │   Bayesian Optimization     │
    └────────────┬───────────────┘
                 │
    ┌────────────▼───────────────┐
    │  Visualizing Results        │
    └────────────┬───────────────┘
                 │
    ┌────────────▼───────────────┐
    │  Saving Results as CSV/XLSX │
    └────────────────────────────┘
  </pre>
</div>

<hr>
