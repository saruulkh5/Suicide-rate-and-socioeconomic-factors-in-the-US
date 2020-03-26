# Name: Saruul Khasar
# Section: AD
# Brief description: This module tests the functions
# created in project.py folder.

from test_functions import assert_equals, check_approx_equals
import project
import pandas as pd
import numpy as np


dir_to_data_folder = "/Users/saruul/Desktop/saruul/data"

dir_unemployment = (dir_to_data_folder+"/unemployment.csv")
dir_education = (dir_to_data_folder+"/education.csv")
dir_suicide = (dir_to_data_folder+"/suicide.csv")
dir_gdp = (dir_to_data_folder+"/gdp_growth.csv")


def unemployment():
    """
    Testing unemployment function in project.py
    """
    print('Testing unemployment function')
    df_unemployment = project.unemployment(2015,1016).loc["California",:]
    unemp = df_unemployment.values
    data = np.array([6.2, 5.5]) 
    ser = pd.Series(data, index =[2015, 2016]).values
    print(np.array_equal(unemp, ser))


def education():
    """
    Testing education function in project.py
    """
    print('Testing education function')
    df_education = project.education(dir_education).loc["Wyoming",:]
    educ = df_education.values
    data = np.array([54.9, 56.7, 56.8, 57.3, 26.9]) 
    ser = pd.Series(data, index =[2014, 2015, 2016, 2017, 2018]).values
    print(np.array_equal(educ, ser))
    

def suicide():
    """
    Testing suicide function in project.py
    """
    print('Testing suicide function')
    df_suicide = project.suicide(dir_suicide).loc["New York",:]
    suic = df_suicide.values
    data = np.array([8.1, 7.8, 8.1, 8.1, 8.3]) 
    ser = pd.Series(data, index =[2014, 2015, 2016, 2017, 2018]).values
    print(np.array_equal(suic, ser))


def gdp_growth():
    """
    Testing gdp_growth function in project.py
    """
    print('Testing gdp_growth function')
    df_gdp = project.suicide(dir_gdp).loc["Washington",:]
    gdp = df_gdp.values
    data = np.array([3.6, 4.4, 3.5, 5.2, 5.8]) 
    ser = pd.Series(data, index =[2014, 2015, 2016, 2017, 2018]).values
    print(np.array_equal(gdp, ser))



def main():
    education()
    suicide()
    gdp_growth()

if __name__ == '__main__':
    main()