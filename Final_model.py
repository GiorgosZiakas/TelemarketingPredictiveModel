import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier  
from sklearn.feature_selection import SelectFromModel   
from sklearn.preprocessing import  StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, get_scorer
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules as arules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from collections import Counter
 





file_path = '/Users/giorgosziakas/Desktop/bank.csv'
bank_data = pd.read_csv(file_path, sep=';')


def load_path(file_path):
    bank_data = pd.read_csv(file_path, sep=';')
    print(bank_data.dtypes)
    print(bank_data.head())
    print(bank_data.info()) 
    print(bank_data.columns)
    print(bank_data.describe(include='all'))
    return bank_data

# Identify the categorical and numerical variables
def identify_features(bank_data):
    categorical_variables = bank_data.select_dtypes(include=['object']).columns
    print('Categorical variables: ', categorical_variables)
    numerical_variables = bank_data.select_dtypes(include=['int64', 'float64']).columns
    print('Numerical variables: ', numerical_variables)


# function for descriptive statistics
def descriptive_stats(bank_data):
    descriptive_stats = bank_data.describe(include = 'all')
    print(descriptive_stats)
    
# Function to analyze the 'y' column
def analyze_y_column(bank_data):
    count = bank_data['y'].value_counts()
    percentage = bank_data['y'].value_counts(normalize=True) * 100
    analysis = pd.DataFrame({'Count': count, 'Percentage': percentage})
    print(analysis)

    
    
# functions for visualizations
def plot_scatter_age_balance(bank_data):
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='age', y='balance', data=bank_data, alpha=0.6)
    plt.title('Age vs Balance')
    plt.xlabel('Age')
    plt.ylabel('Balance')
    plt.show() 
    
 
    
def plot_boxplot_balance(bank_data):
    plt.figure(figsize=(10,6)) 
    sns.boxplot(y=bank_data['balance']) 
    plt.title('Boxplot of Balance') 
    plt.ylabel('Balance')  
    plt.show()
   
    
    
def plot_histogram_duration(bank_data):
    plt.figure(figsize=(10,6))  
    sns.histplot(bank_data['duration'], kde=True)
    plt.title('Histogram of Call Duration')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.show()
    
def plot_bar_chart_job(bank_data):
    plt.figure(figsize=(10,6))
    sns.countplot(x='job', data=bank_data, palette='Set2') 
    plt.title('Count of Job Categories')   
    plt.xlabel('Job Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()
    
def plot_bar_chart_marital(bank_data):
    plt.figure(figsize=(10,6))  
    sns.countplot(x='marital', data=bank_data)
    plt.title('Bar Chart of Marital Status')
    plt.xlabel('Marital Status')
    plt.ylabel('Frequency')
    plt.show()
    
def plot_bar_chart_education(bank_data):
    plt.figure(figsize=(10,6))  
    sns.countplot(x='education', data=bank_data)
    plt.title('Bar Chart of Education')
    plt.xlabel('Education')
    plt.ylabel('Frequency')
    plt.show()
    
def plot_pair_plot(bank_data):
    numerical_subset = ['age', 'balance', 'duration', 'campaign']
    sns.pairplot(bank_data[numerical_subset])
    plt.suptitle('Pair Plot of Selected Numerical Variables', y=1.02)
    plt.show()
    

    
# Distribution of age
def plot_distribution_age(bank_data):
    plt.figure(figsize=(10,6))
    sns.histplot(bank_data['age'], kde=True)
    plt.title('Histogram of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()
    
    
# Distribution of Responses
def response_distribution(bank_data):
    plt.figure(figsize=(10,6))  
    sns.countplot(x='y', data=bank_data)
    plt.title('Distribution of Responses')
    plt.xlabel('Response')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.show()
    
    
# Box plot for numerical features against a categorical feature (Campaign)
def plot_box_for_categorical(bank_data, categorical_feature, numerical_feature):
    for feature in numerical_feature:
        plt.figure(figsize=(10,6))
        sns.boxplot(x=categorical_feature, y=feature, data=bank_data)
        plt.title('Box Plot of ' + feature)
        plt.xlabel(categorical_feature)
        plt.ylabel(feature)
        plt.show()
        
# Bar plots for the target variable against a categorical feature
def barplot_categorical_target(bank_data, categorical_feature):
    plt
    sns.countplot(x=categorical_feature, hue='y', data=bank_data)
    plt.title('Bar Plot of ' + categorical_feature)
    plt.xticks(rotation=45) 
    plt.show()
    
def plot_boxplot_duration_response(bank_data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='y', y='duration', data=bank_data)
    plt.title('Boxplot of Call Duration by Response')
    plt.xlabel('Response')
    plt.ylabel('Duration (seconds)')
    plt.show()
    
    
def correlation_matrix(bank_data):
    
    # Convert target variable 'y' to numeric for correlation analysis.
    bank_data['y_numeric'] = bank_data['y'].apply(lambda x: 1 if x == 'yes' else 0)
    
    # Select only numeric columns for correlation matrix
    numeric_columns = bank_data.select_dtypes(include=['number'])
    
    # Correlation matrix 
    correlation_matrix = numeric_columns.corr()   
    
    # Plotting the correlation matrix   
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    
    # Correlation of numerical features with the target variable 'y'
    correlation_with_target = correlation_matrix['y_numeric'].sort_values(ascending=False)
    correlation_with_target.drop('y_numeric', inplace=True)  # Drop self-correlation
    return correlation_with_target

def plot_duration_y(bank_data): 
    
    # Convert target variable 'y' to numeric.
    bank_data['y_numeric'] = bank_data['y'].apply(lambda x: 1 if x == 'yes' else 0)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='y', y='duration', data=bank_data)
    plt.title('Boxplot of Call Duration by Response')
    plt.xlabel('Response')
    plt.ylabel('Duration (seconds)')
    plt.show()  


# Function to plot distribution of target variable across categorical features
def plot_categorical_distribution(data, column, target):
    plt.figure(figsize=(10, 5))
    sns.countplot(x=column, hue=target, data=data)
    plt.title(f"Distribution of {target} across {column}")
    plt.xticks(rotation=45)
    plt.show()

# Analyzing the impact of some key categorical features on the target variable
for column in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'campaign']:
    plot_categorical_distribution(bank_data, column, 'y')
    
# Creatin a plot for the binned 'campaign' feature vs 'y'   
def plot_binned_campaign(bank_data):
    # Let's start by creating the binned 'campaign' feature as recommended.
    campaign_bins = [-1, 3, 6, 9, float('inf')]  # Define bins
    campaign_labels = ['1-3', '4-6', '7-9', '>9']
    bank_data['campaign_binned'] = pd.cut(bank_data['campaign'], bins=campaign_bins, labels=campaign_labels)

    # Now, let's visualize the new binned feature in relation to the target variable 'y'
    plt.figure(figsize=(10, 6))
    sns.countplot(x='campaign_binned', hue='y', data=bank_data)
    plt.title("Binned Campaign vs Target Variable ('y')")
    plt.show()
    
# Also create a function to run all visualizations at once
def run_all_visualizations(bank_data):
    plot_scatter_age_balance(bank_data)
    plot_boxplot_balance(bank_data)
    plot_histogram_duration(bank_data)
    plot_bar_chart_job(bank_data)
    plot_bar_chart_marital(bank_data)
    plot_bar_chart_education(bank_data)
    plot_pair_plot(bank_data)
    plot_distribution_age(bank_data)
    response_distribution(bank_data)
    plot_box_for_categorical(bank_data, 'job', ['balance', 'duration', 'campaign'])
    plot_box_for_categorical(bank_data, 'poutcome', ['balance', 'campaign'])
    barplot_categorical_target(bank_data, 'month')
    plot_boxplot_duration_response(bank_data)
    correlation_matrix(bank_data)
    plot_duration_y(bank_data) 
    plot_categorical_distribution(bank_data, 'job', 'y')
    plot_binned_campaign(bank_data) 
   
   
def preprocess_data(bank_data): 
    X = bank_data.drop(['y'], axis=1)  # Features
    y = bank_data['y']  # Target
    
    # Drop 'duration' column
    bank_data = bank_data.drop(['duration'], axis=1)
    
    # Ensure 'balance' doesn't contain negative or zero values
    bank_data['balance'] = bank_data['balance'].clip(lower=1)
    
     # Handle outliers in 'balance' by capping
    upper_limit = bank_data['balance'].quantile(0.95)
    bank_data['balance'] = np.where(bank_data['balance'] > upper_limit, upper_limit, bank_data['balance'])
    
    # Apply a logarithmic transformation to 'balance'
    bank_data['log_balance'] = np.log(bank_data['balance'])
    
    # Binary encoding for high-response months
    high_response_months = ['may', 'jun', 'jul', 'aug']
    bank_data['high_response_month'] = bank_data['month'].isin(high_response_months).astype(int)
    
    X = X.drop(['month', 'balance'], axis=1)
   
    # Encoding the target variable 'y'
    y_encoded = LabelEncoder().fit_transform(y)
    
    
    # Identifying categorical and numerical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns

    # Creating column transformer for both categorical and numerical operations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(), categorical_columns)
        ])

    # Creating a pipeline with the preprocessor
    prep_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Applying the pipeline to the feature set X
    X_prepared = prep_pipeline.fit_transform(X)
    

    return X_prepared, y_encoded, preprocessor
    
    



def cross_val_feature_selection(X, y, preprocessor, estimator, cv=5, n_features_to_select=None, scoring='accuracy'):
    cv = StratifiedKFold(n_splits=cv)
    feature_importances = np.zeros(X.shape[1])
    scores = []
    selected_features_list = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Fit the feature selection model on the training set
        estimator.fit(X_train, y_train)
        model = SelectFromModel(estimator, prefit=True, max_features=n_features_to_select)
        X_train_selected = model.transform(X_train)
        X_test_selected = model.transform(X_test)

        # Train the downstream model on the selected features
        downstream_model = clone(estimator)  # Clone the estimator to make sure we have a fresh model
        downstream_model.fit(X_train_selected, y_train)

        # Score the model on the test set using the provided scoring metric
        scorer = get_scorer(scoring)
        score = scorer(downstream_model, X_test_selected, y_test)
        scores.append(score)

        # Get feature importances
        feature_importances += estimator.feature_importances_

        # Record the selected features
        selected_features = preprocessor.get_feature_names_out()[model.get_support()]
        selected_features_list.append(selected_features)

    # Calculate average feature importances and scores
    feature_importances /= cv.get_n_splits()
    average_score = np.mean(scores)

    # Identify the most frequently selected features
    all_selected_features = np.concatenate(selected_features_list)
    feature_counts = Counter(all_selected_features)
    most_common_features = feature_counts.most_common(n_features_to_select)

    return most_common_features, average_score

# Clustering function
def perform_evaluate_clustering(bank_data):
    
    # Applying a logarithmic transformation to 'balance' to reduce skewness
    bank_data['log_balance'] = np.log(bank_data['balance'] + 1)
    
    # Replace NaN AND -inf values in 'log_balance' with mean
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    bank_data['log_balance'] = imputer.fit_transform(bank_data['log_balance'].replace(-np.inf, np.nan).values.reshape(-1, 1))
    
    # Selecting numerical features for clustering
    num_features_log = ['age', 'log_balance', 'day', 'campaign', 'pdays', 'previous']
    num_data_log = bank_data[num_features_log]
    
    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(num_data_log)
    
    # Clustering algorithms
    # Using KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
    print(f'KMeans Silhouette Score: {kmeans_silhouette:.2f}')
    
    # Using DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)   
    dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
    print(f'DBSCAN Silhouette Score: {dbscan_silhouette:.2f}')
    
    # Using Agglomerative Clustering
    agglomerative = AgglomerativeClustering(n_clusters=3)
    agglomerative_labels = agglomerative.fit_predict(X_scaled) 
    agglomerative_silhouette = silhouette_score(X_scaled, agglomerative_labels)
    print(f'Agglomerative Clustering Silhouette Score: {agglomerative_silhouette:.2f}')
    
    # Using Spectral Clustering
    spectral = SpectralClustering(n_clusters=3, random_state=42, affinity='nearest_neighbors')
    spectral_labels = spectral.fit_predict(X_scaled)  
    spectral_silhouette = silhouette_score(X_scaled, spectral_labels)
    print(f'Spectral Clustering Silhouette Score: {spectral_silhouette:.2f}')    
    
    # Visualize the clusters with PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)  
    
    # Visualization
    plt.figure(figsize=(20, 5))
    clustering_labels = [kmeans_labels, dbscan_labels, agglomerative_labels, spectral_labels]
    titles = ['KMeans', 'DBSCAN', 'Agglomerative', 'Spectral']
    
    for idx, labels in enumerate(clustering_labels):
        plt.subplot(1, 4, idx + 1)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
        plt.title(titles[idx])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        
    plt.tight_layout()
    plt.show()
        
    return kmeans_labels, dbscan_labels, agglomerative_labels, spectral_labels  

# Function for association rules   
def find_association_rules(bank_data, min_support=0.05, confidence_threshold=0.8):   
    
    # Define Bins for numerical features
    numerical_bins = {
        'age': {'bins': [0, 30, 60, 100], 'labels': ['Young', 'Senior', 'Elderly']},
        'balance': {'bins': [-float('inf'), 0, 1000, 5000, float('inf')], 'labels': ['Negative', 'Low', 'Medium', 'High']},
        'duration': {'bins': [0, 100, 300, 600, float('inf')], 'labels': ['Short', 'Medium', 'Long', 'Very Long']},
        'campaign': {'bins': [0, 2, 5, 50], 'labels': ['Low', 'Medium', 'High']},
        'pdays': {'bins': [-1, 0, 100, float('inf')], 'labels': ['Not Contacted', 'Recently Contacted', 'Long Ago']},
        'previous': {'bins': [0, 1, 3, 25], 'labels': ['None', 'Few', 'Many']}
    }
    
    # Discretize numerical features
    for col, bins in numerical_bins.items():
        bank_data[col + '_binned'] = pd.cut(bank_data[col], bins=bins['bins'], labels=bins['labels'])   
        
    # Selecting categorical features and new binned features
    categorical_columns = bank_data.select_dtypes(include = ['object']).columns.tolist()
    binned_columns = [col + '_binned' for col in numerical_bins]
    selected_columns = categorical_columns + binned_columns
    
    # One-hot encoding the selected features
    transactions = bank_data[selected_columns].applymap(str).values.tolist()
    te = TransactionEncoder()   
    te_array = te.fit_transform(transactions)
    df_transactions = pd.DataFrame(te_array, columns=te.columns_)   
    
    
    # Apply Aprioro algorithm
    frequent_itemsets = apriori(df_transactions, min_support=0.05, use_colnames=True)
    
    # Generate association rules
    rules = arules(frequent_itemsets, metric='confidence', min_threshold=confidence_threshold)
    
    # Filter rules based on the confidence
    rules = rules[rules['confidence'] >= confidence_threshold]
    
    return rules

# Function for Logistic Regression model
def train_logistic_regression(X_train, y_train):
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)
    return log_reg
    


# Function for Decision Tree model
def train_decision_tree(X_train, y_train):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    return dt

# Function for Random Forest model
def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')    
    rf_model.fit(X_train, y_train)
    return rf_model

# Function for XGBoost model
def train_xgboost(X_train, y_train):
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', 
    random_state=42, scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train))
    xgb_model.fit(X_train, y_train)
    return xgb_model



# Function for MLPclassifier model
def train_mlp_classifier(X_train, y_train, hidden_layer_sizes=(100,), max_iter=1000, learning_rate_init=0.1, random_state=42):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, learning_rate_init=learning_rate_init, random_state=random_state)
    mlp.fit(X_train, y_train)
    return mlp



# Function for evaluating the models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores: {cv_scores.mean():.2f}(+/- {cv_scores.std() * 2:.2f})")
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    print(f'Classification Report: \n {report}')
    conf_matrix = confusion_matrix(y_test, predictions)
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.show()
    



# Create main function of the programm in order to call all the functions
def main():
    
    # Load the data
    file_path = '/Users/giorgosziakas/Desktop/bank.csv'
    bank_data = load_path(file_path)  
    
    # Identify the categorical and numerical variables
    identify_features(bank_data)
    
    # Descriptive statistics
    descriptive_stats(bank_data)
    
    # Check for missing values
    if bank_data.isnull().sum().any():
        print('There are missing values in the dataset')
        
    # 'y' column analysis
    analyze_y_column(bank_data)
    
    #  Visualizations
    run_all_visualizations(bank_data)
    
   
    # Preprocessing
    X_prepared, y_encoded, preprocessor= preprocess_data(bank_data) 
    # Checking the target variable 'y' and shape of X
    print("Unique values in y_encoded:", np.unique(y_encoded))
    print("Data type of y_encoded:", y_encoded.dtype)  
    print("Shape of X after preprocessing:", X_prepared.shape)
    
    # Clustering
    perform_evaluate_clustering(bank_data)
    kmeans_labels, dbscan_labels, agglomerative_labels, spectral_labels  = perform_evaluate_clustering(bank_data)
    
    # Visualize the clusters with numerical features (only Agglomerative cluster with highest silhouette score)
    num_features = ['age', 'log_balance', 'day', 'campaign', 'pdays', 'previous']
    for feature in num_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=agglomerative_labels, y=bank_data[feature])
        plt.title(f'{feature} by Agglomerative Clusters')
        plt.xlabel('Agglomerative Clusters')
        plt.ylabel(feature)
        plt.show()
        
    # Association Rules
    rules = find_association_rules(bank_data)
    print(rules.sort_values(by='confidence', ascending=False))
    
    
    # Feature Selection
    clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
    top_features, avg_score = cross_val_feature_selection(X_prepared, y_encoded, preprocessor, clf, cv=5, n_features_to_select=6, scoring='f1_weighted')
    print("Top features selected during cross-validation:", top_features)
    print("Average F1 score across folds:", avg_score)
    
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_prepared, y_encoded, test_size=0.2, random_state=42)
    # Add a print statement here to confirm the shape after splitting   
    print("Shape of X_train:", X_train.shape)
    
    
 
    # Train and compare the models
    log_reg_model = train_logistic_regression(X_train, y_train)
   
    dt_model = train_decision_tree(X_train, y_train)
    
    rf_model = train_random_forest(X_train, y_train)
    
    xgb_model = train_xgboost(X_train, y_train)
    
    best_mlp_model = train_mlp_classifier(X_train, y_train,hidden_layer_sizes=(100, 50, 25))
    # Evaluate the models
    
    print('Evaluation of Logistic Regression Model')  
    evaluate_model(log_reg_model, X_train, X_test, y_train, y_test)
       
    print('Evaluation of Decision Tree Model')  
    evaluate_model(dt_model, X_train, X_test, y_train, y_test)
    
    print('Evaluation of Random Forest Model')
    evaluate_model(rf_model, X_train, X_test, y_train, y_test)
    
    print('Evaluation of XGBoost Model')
    evaluate_model(xgb_model, X_train, X_test, y_train, y_test)

    print('Evaluation of MLP Model')
    evaluate_model(best_mlp_model, X_train, X_test, y_train, y_test)  
    
    
if __name__ == '__main__':
    main()
    
    

