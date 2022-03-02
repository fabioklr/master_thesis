# Load Forbes Global 2000 data
companies = pd.read_csv("data/Forbes Global 2000 - 2019.csv")
companies = companies[companies['Continent'] == 'Europe']
print((companies.iloc[:5]).to_markdown())