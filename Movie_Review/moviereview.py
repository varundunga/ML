import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
home = str(Path.home())
movies_header=['MovieID','Title','Genres']
movies=pd.read_csv(home + '\\Machine learning\\Movie_Review\\movies.dat',sep='::',header=None,names=movies_header)
users_header=['UserID','Gender','Age','Occupation','Zip-code']
users=pd.read_csv(home + '\\Machine learning\\Movie_Review\\users.dat',sep='::',header=None,names=users_header)
ratings_header=['UserID','MovieID','Rating','Timestamp']
ratings=pd.read_csv(home + '\\Machine learning\\Movie_Review\\ratings.dat',sep='::',header=None,names=ratings_header)
