# %%
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split

# %%
from model_src.baseline_models import RS_baseline, RS_baseline_usr_mov

# %%
prefix = "https://github.com/mlops-itba/EDA/"
scores_url = prefix + "raw/main/data/scores_0.csv"
peliculas_url = prefix + "raw/main/data/peliculas_0.csv"
personas_url = prefix + "raw/main/data/personas_0.csv"
trabajadores_url = prefix + "raw/main/data/trabajadores_0.csv"
usuarios_url = prefix + "raw/main/data/usuarios_0.csv"

df_scores = pd.read_csv(scores_url)

df_scores_train, df_scores_test = train_test_split(df_scores, test_size = 0.2)

# %%
model = RS_baseline()
model.fit(df_scores_train)
model.score(df_scores_test[["user_id", "movie_id"]].values,df_scores_test["rating"].values)
model.save_model("baseline_mean.pkl")

# %%
model = RS_baseline_usr_mov(0.0)
model.fit(df_scores_train)
model.score(df_scores_test[["user_id", "movie_id"]].values,df_scores_test["rating"].values)
model.save_model("baseline_p_0.0.pkl")

# %%
model = RS_baseline_usr_mov(1.0)
model.fit(df_scores_train)
model.score(df_scores_test[["user_id", "movie_id"]].values,df_scores_test["rating"].values)
model.save_model("baseline_p_1.0.pkl")
# %%
model = RS_baseline_usr_mov(0.5)
model.fit(df_scores_train)
model.score(df_scores_test[["user_id", "movie_id"]].values,df_scores_test["rating"].values)
model.save_model("baseline_p_0.5.pkl")

# %%
