import numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

class TitanicAnalysis():
  def __init__(self, df):
    self.df = df
  def clean(self):
    self.df['Age'] = self.df['Age'].fillna(self.df['Age'].mean())
    self.df = self.df.drop(columns=['Cabin'])
    self.df = self.df.dropna(subset=['Embarked'])
    return self.df
  def survival_by_sex(self):
    survi = self.df.groupby('Sex')['Survived'].mean()
    return survi
  def survival_by_class(self):
    cl = self.df.groupby('Pclass')['Survived'].mean()
    return cl
  def average_age_by_sex(self):
    age = self.df.groupby('Sex')['Age'].mean()
    return age
hod = TitanicAnalysis(df)
print(hod.clean())
print(hod.survival_by_sex())
print(hod.survival_by_class())
print(hod.average_age_by_sex())
