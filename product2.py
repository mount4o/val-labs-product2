import streamlit as st
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from geopy.distance import geodesic

# Load the KNN model
with open("saved_models/knn_public_companies.sav", "rb") as model_file:
    knn_model = pickle.load(model_file)

# Load your dataset (assuming it's the same one used to train the model)
ready_for_knn_df = pd.read_csv("data/per_sector_ready_for_knn_stock_data(2).csv", index_col='Symbol')
companies_df = pd.read_csv("data/per_sector_readable_data(2).csv", index_col='Symbol')

# Drop 'Market Cap' from the ready_for_knn_df as it should not be scaled
ready_for_knn_df = ready_for_knn_df.drop(columns=['Market Cap'])

# Extract features and company names
company_names = companies_df['Name']
country_options = companies_df['Country'].dropna().unique().tolist()
industry_options = companies_df['Industry'].dropna().unique().tolist()
 # as the most near neighbor of the company will be itself we'll always ignore the first neighbor, so we expect n_neighbors - 1 results
n_neighbors = 6

# Initialize the scaler with the data used for training
scaler = StandardScaler().fit(ready_for_knn_df)

# Country MDS encoding
countries_coords = {
    'Singapore': (1.3521, 103.8198),
    'France': (46.603354, 1.888334),
    'United States': (37.0902, -95.7129),
    'Hong Kong': (22.3193, 114.1694),
    'Belgium': (50.8503, 4.3517),
    'China': (35.8617, 104.1954),
    'Ireland': (53.1424, -7.6921),
    'Netherlands': (52.1326, 5.2913),
    'Australia': (-25.2744, 133.7751),
    'United Kingdom': (55.3781, -3.4360),
    'Canada': (56.1304, -106.3468),
    'Israel': (31.0461, 34.8516),
    'Malaysia': (4.2105, 101.9758),
    'Japan': (36.2048, 138.2529),
    'Germany': (51.1657, 10.4515),
    'Cayman Islands': (19.3133, -81.2546),
    'Spain': (40.4637, -3.7492),
    'Denmark': (56.2639, 9.5018),
    'Switzerland': (46.8182, 8.2275),
    'Taiwan': (23.6978, 120.9605),
    'Indonesia': (-0.7893, 113.9213),
    'South Korea': (35.9078, 127.7669)
}

# Initialize a distance matrix
distance_matrix = pd.DataFrame(index=countries_coords.keys(), columns=countries_coords.keys())

# Calculate the distance between each pair of countries
for country1 in countries_coords.keys():
    for country2 in countries_coords.keys():
        distance_matrix.loc[country1, country2] = geodesic(countries_coords[country1], countries_coords[country2]).kilometers

distance_matrix = distance_matrix.astype(float)

mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
country_coords_mds = mds.fit_transform(distance_matrix)

# Convert the MDS result to a DataFrame
mds_df = pd.DataFrame(country_coords_mds, index=distance_matrix.index, columns=['MDS1', 'MDS2'])

# Streamlit app
st.title("Find Similar Public Companies")

# Option to select a company from the dataset or enter custom data
option = st.radio("Choose an option:", ("Select from existing companies", "Enter custom company data"))

if option == "Select from existing companies":
    selected_company = st.selectbox("Select a company", company_names)
    
    if st.button("Find Nearest Company"):
        try:
            # Find the symbol of the selected company
            company_symbol = companies_df[companies_df['Name'] == selected_company].index[0]
            
            # Ensure that the symbol is treated as a standard Python string
            company_symbol = str(company_symbol)
            
            # Get the features of the selected company from companies_df
            selected_company_details = companies_df.loc[company_symbol].drop('Market Cap', errors='ignore')
            
            # Display the selected company's details
            st.write(f"**Features of the selected company {selected_company}**")
            st.write(selected_company_details.to_frame().T)  # Display the details in a table format
            st.write(f"**The selected company's market cap is {companies_df.loc[company_symbol]['Market Cap']}M**")
            
            # Get the input features of the selected company from ready_for_knn_df
            input_data = ready_for_knn_df.loc[[company_symbol]]

            # Scale the input data
            input_data_scaled = scaler.transform(input_data)

            # Find the nearest neighbors
            distances, indices = knn_model.kneighbors(input_data_scaled, n_neighbors=n_neighbors)

            # Display the results, skipping the first result
            st.write(f"**Nearest companies to {selected_company}**")
            for i in range(1, n_neighbors):  # Start from 1 to skip the first (duplicate) result
                nearest_company_symbol = ready_for_knn_df.index[indices.flatten()[i]]
                nearest_company_details = companies_df.loc[nearest_company_symbol]
                
                # Display all features of the nearest company
                st.write(f"{i}. **{nearest_company_details['Name']}**")
                st.write(nearest_company_details.to_frame().T)  # Display the details in a table forma

        except Exception as e:
            st.error(f"An error occurred: {e}")
elif option == "Enter custom company data":
    # Create a form for user input
    with st.form("company_form"):
        name = st.text_input("Company Name")
        symbol = st.text_input("Company Symbol")
        sector = st.selectbox("Sector", companies_df['Sector'].dropna().unique().tolist())
        industry = st.selectbox("Industry", industry_options)
        full_time_employees = st.number_input("Full Time Employees", min_value=0, step=1)
        country = st.selectbox("Country", country_options)
        ebtida = st.number_input("EBTIDA (in millions)", format="%.6f", step=0.01)
        total_revenue = st.number_input("Total Revenue (in millions)", format="%.6f", step=0.01)
        ev_revenue = st.number_input("EV / Revenue", format="%.6f", step=0.01)
        ev_ebitda = st.number_input("EV / EBITDA", format="%.6f", step=0.01)

        # Form submission
        submitted = st.form_submit_button("Find Nearest Company")

    if submitted:
        try:
            industry_mean_market_cap = companies_df[companies_df['Industry'] == industry]['Market Cap'].mean()
            # sector_mean_market_cap = companies_df[companies_df['Sector'] == sector]['Market Cap'].mean()

            # Convert country to MDS coordinates
            country_coords = mds_df.loc[country]
            country_mds1, country_mds2 = country_coords['MDS1'], country_coords['MDS2']

            # Prepare the input features as a DataFrame
            input_data = pd.DataFrame({
                'Full Time Employees': [full_time_employees],
                'EBTIDA': [ebtida],
                'Total Revenue': [total_revenue],
                'EV / Revenue': [ev_revenue],
                'EV / EBITDA': [ev_ebitda],
                # 'Sector Encoded': [sector_mean_market_cap],
                'Industry Encoded': [industry_mean_market_cap],
                'MDS1': [country_mds1],
                'MDS2': [country_mds2]
            }, index=[symbol])

            # Drop 'Market Cap' from the input data to match the training features
            input_data = input_data.drop(columns=['Market Cap'], errors='ignore')

            # st.write(f" Input data columns: **{set(input_data.columns)}**, Required model columns: **{set(ready_for_knn_df.columns)}**")
            # Ensure the input data columns match those expected by the model
            if set(input_data.columns) != set(ready_for_knn_df.columns):
                st.error(f"Input data columns do not match the model's expected input. Please check the input feature names. Input data columns: **{set(input_data.columns)}**, Required model columns: **{set(ready_for_knn_df.columns)}**")
            else:
                # Scale the input data
                input_data_scaled = scaler.transform(input_data)

                # Find the nearest neighbors
                distances, indices = knn_model.kneighbors(input_data_scaled, n_neighbors=n_neighbors)

                # Display the input company details (excluding 'Market Cap')
                st.write(f"**Features of the entered company ({name})**")
                st.write(pd.DataFrame({
                    'Company Name': [name],
                    'Sector': [sector],
                    'Industry': [industry],
                    'Full Time Employees': [full_time_employees],
                    'Country': [country],
                    'EBTIDA (in millions)': [ebtida],
                    'Total Revenue (in millions)': [total_revenue],
                    'EV / Revenue': [ev_revenue],
                    'EV / EBITDA': [ev_ebitda]
                }, index=[symbol]))

                # Display the results, skipping the first result
                st.write(f"**Nearest companies to {name} ({symbol})**")
                for i in range(1, n_neighbors):  # Start from 1 to skip the first (potential duplicate) result
                    nearest_company_symbol = ready_for_knn_df.index[indices.flatten()[i]]
                    nearest_company_details = companies_df.loc[nearest_company_symbol]

                    # Display all features of the nearest company
                    st.write(f"{i}. **{nearest_company_details['Name']}**")
                    st.write(nearest_company_details.to_frame().T)  # Display all the details in a table format

        except Exception as e:
            st.error(f"An error occurred: {e}")
