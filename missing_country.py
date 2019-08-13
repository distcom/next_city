import json
import pandas as p
from iso3166 import countries
import pandas as pd
import numpy as np


def load_data_as_df(file):
    
    with open(file) as train_file:
        dict_train = json.load(train_file)

    dframe = pd.DataFrame(dict_train)
    
    return dframe
    
def preprocess_data(dframe):
    
    dframe["user"] = dframe["user"].apply(lambda x: x[0][0])
    dframe["session_id"] = dframe["session_id"].apply(lambda x: x[0])
    dframe["unix_timestamp"] = dframe["unix_timestamp"].apply(lambda x: x[0])
    
    return dframe
    

def find_user_countries_set(dframe): 
    seri = dframe["user"].apply(lambda x: x.get('country', 0))
    countriesSet = set(seri.unique())
    
    return countriesSet


def get_cities_users_unknown(dframe):
    
    # split list of cities searched by users (with missed country), if country of user is known, replace it with None
    citiesSeries = dframe.apply(lambda x: x["cities"][0].split(', ') if x["user"]['country'] in [""] else None, axis=1)

    # drop those rows with None values
    citiesSeries = citiesSeries.dropna()
    
    # returns a series of list of cities for users with unknown country
    return citiesSeries



def get_counts_of_cities_searched(citiesSeries):
    
    # Count cities, searched. Sort counts from highest to lowest 
    countsOfCitiesSearchedSeries = citiesSeries.apply(pd.Series).stack().value_counts()
    
    # Rename the columns
    dfcities = countsOfCitiesSearchedSeries.rename_axis('city').reset_index(name='count')
    
    return dfcities


def get_country_of_cities(world_cities_csv):
    
    # dataframe read from the csv file; Cities & Countries
    citiesCountriesDF = pd.read_csv(world_cities_csv, sep=',')
    # update dataframe to remove accents
    citiesCountriesDF["name"] = citiesCountriesDF["name"].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    
    return citiesCountriesDF


def countryToAbr(x):
  
  if x not in ['United Kingdom', 'Russia', '']:
    return countries.get(x).alpha2.upper()
  elif x == 'United Kingdom':
    return 'UK'
  elif x == 'Russia':
    return 'RU'
  else:
    return x


def replace_missing_country_for_users(userSeries, missing_country):
    
    for x in userSeries:
        if x["country"] in [""]:
            x["country"] = missing_country
            
    
    return userSeries


def country_to_freq_of_cities(dfcities, citiesCountriesDF):
    
  for row in dfcities.itertuples():
      city_name = row.city.rsplit(' ', 1)[0]  # removing the state/province from the end of the list
  
      citydf = citiesCountriesDF.loc[citiesCountriesDF["name"].str.lower() == city_name.lower()]
      if len(citydf["country"]) == 1:
        dfcities.at[row.Index, 'country'] = citydf["country"].iloc[0]
        # print(vdf["country"].iloc[0])
    
      elif len(citydf["country"]) > 1:    # this city  belongs to multiple countries, add the one which is in the list. More advanced fixes may be required if same city belong in the list of our countries
        dfcities.at[row.Index, 'country'] = citydf["country"].iloc[0]
      else:
        dfcities.at[row.Index, 'country'] = ""
        # print("{} \t No Value Found----------------------------------------------------------------------------------------------------------------".format(city_name))

  return dfcities


def calculate_probabilities_for_missing_county(dfcities, countriesSet):
    # group rows by 'cn' column, and sum up
    searchedCitiesTotalDf = dfcities.groupby(['cn'], as_index=False).sum()
    # sort values by count
    searchedCitiesTotalDf = searchedCitiesTotalDf.sort_values(by=['count'], ascending=False)
    # For calculating probabilities, only consider countries which are missing in the users' countries
    missingCountriesdf = searchedCitiesTotalDf[~searchedCitiesTotalDf["cn"].isin(countriesSet)]

    # sum up the counts
    totalSearchCountMissing = missingCountriesdf["count"].sum()
    
    # calculate probability
    missingCountriesdf["probability"] = missingCountriesdf["count"]/totalSearchCountMissing

    # generate country column by getting name
    missingCountriesdf["country"] = missingCountriesdf["cn"].apply(lambda x: countries.get(x).name)
    
    # get only columns we are interested in
    missingCountriesdf = missingCountriesdf[['cn', 'country', 'count', 'probability']]
    
    return missingCountriesdf
    
    

def missining_country_rates(json_file, world_cities_csv):
    
    # load the JSON data
    dframe = load_data_as_df(json_file)
    
    # Clean up data to remove lists in the columns
    preprocess_data(dframe)
    
    # find list of users' countries
    countriesSet = find_user_countries_set(dframe)
    
    # split list of cities searched by users (with missed country), if country of user is known, drop it!
    citiesSeries = get_cities_users_unknown(dframe)
    
    # count the search counties
    dfcities = get_counts_of_cities_searched(citiesSeries)
    
    # dataframe read from the csv file; Cities & Countries
    citiesCountriesDF = get_country_of_cities(world_cities_csv)
    
    # add country to frequency of cities
    country_to_freq_of_cities(dfcities, citiesCountriesDF)
    
    #create a column for countries of 2-letter abbreviations
    dfcities["cn"] = dfcities["country"].apply(countryToAbr)
    
    # count countries 
    countrySearchedCounts = dfcities["cn"].apply(pd.Series).stack().value_counts()
    
    # Generate the probabilities
    missingCountriesdf = calculate_probabilities_for_missing_county(dfcities, countriesSet)
    
    # return the generated probabilities for different countries, as missing country. 
    return missingCountriesdf


def missining_country_likely(json_file, world_cities_csv):
    
    # Generate the missing country probabilities dataframe
    missingCountriesdf = missining_country_rates(json_file, world_cities_csv)
    
    # return the most likely country
    return missingCountriesdf.iloc[0]
