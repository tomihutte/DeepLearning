import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import graphviz as gv

sns.set(style='whitegrid')

# Cargo los datos
data = pd.read_csv("carseats.csv", delimiter=",")
# Creo la variable High
data["High"] = [
    1 if data["Sales"][i] >= 8 else 0 for i in range(len(data["Sales"]))
]

# Reemplazo las palabras por numero para que el arbol lo maneje
data.replace("Yes", 1, inplace=True)
data.replace("No", 0, inplace=True)
data.replace("Good", 2, inplace=True)
data.replace("Medium", 1, inplace=True)
data.replace("Bad", 1, inplace=True)

# Hago la division de datos
# np.random.seed(10)
t_size = 0.3

train, test = train_test_split(data, test_size=t_size, stratify=data["High"])
# train, val = train_test_split(train, test_size=v_size, random_state=random_state, stratify=train["High"])

##### Inciso B #####

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# Dropeo los datos de sales y separo en clasificacion y parametros

x_train, y_train = train.drop(["High", "Sales"], axis=1), train["High"]
# x_val, y_val = val.drop(["High", "Sales"], axis=1), val["High"]
x_test, y_test = test.drop(["High", "Sales"], axis=1), test["High"]

TreeC = DecisionTreeClassifier()
TreeC.fit(x_train, y_train)

dot_data = tree.export_graphviz(
    TreeC,
    leaves_parallel=False,
    label="all",
    out_file=None,
    filled=True,
    rounded=True,
    special_characters=True,
    rotate=True,
)
graph = gv.Source(dot_data)
graph.render("Class_Tree(b)")

print("--------Inciso B - Clasificador--------")
print("Precisión sobre train: {}".format(TreeC.score(x_train, y_train)))
print("Precisión sobre test: {}".format(TreeC.score(x_test, y_test)))

##### Inciso C #####
from sklearn.tree import DecisionTreeRegressor

x_train, y_train = train.drop(["High", "Sales"], axis=1), train["Sales"]
# x_val, y_val = val.drop(["High", "Sales"], axis=1), val["High"]
x_test, y_test = test.drop(["High", "Sales"], axis=1), test["Sales"]

TreeR = DecisionTreeRegressor()
TreeR.fit(x_train, y_train)

dot_data = tree.export_graphviz(
    TreeR,
    leaves_parallel=False,
    label="all",
    out_file=None,
    filled=True,
    rounded=True,
    special_characters=True,
    rotate=True,
)
graph = gv.Source(dot_data)
graph.render("Class_Tree(c)")

print("--------Inciso C - Regresor--------")
print("Score sobre train: {}".format(TreeR.score(x_train, y_train)))
print("Score sobre test: {}".format(TreeR.score(x_test, y_test)))

#### Inciso D #####
print("--------Inciso D - Comparación--------")
print("MSE Regresion train: {}".format(
    np.mean((TreeR.predict(x_train) - y_train))**2))
print("MSE Regresion test: {}".format(
    np.mean((TreeR.predict(x_test) - y_test))**2))

#### Inciso E #####
from sklearn.model_selection import GridSearchCV as GCV

TreePath = DecisionTreeRegressor()

params = [{
    "max_depth": np.arange(1, 20),
    "ccp_alpha": np.linspace(0, 2, num=100)
}]

g_search = GCV(TreePath,
               param_grid=params,
               cv=5,
               verbose=0,
               scoring='neg_mean_squared_error')

g_search.fit(x_train, y_train)

best_params = g_search.best_params_
best_estimator = g_search.best_estimator_

print("--------Inciso E - CV + Pruning--------")
print(best_params)
print("Score sobre train del mejor estimador conseguido: {}".format(
    best_estimator.score(x_train, y_train)))
print("Score sobre test del mejor estimador conseguido: {}".format(
    best_estimator.score(x_test, y_test)))

dot_data = tree.export_graphviz(
    best_estimator,
    leaves_parallel=False,
    label="all",
    out_file=None,
    filled=True,
    rounded=True,
    special_characters=True,
    rotate=True,
)
graph = gv.Source(dot_data)
graph.render("Class_Tree(e)")

#### Inciso F #####

from sklearn.ensemble import BaggingRegressor

TreePruned = best_estimator  ## voy a usar el ultimo estimador para hacer el Bagging

BaggingTree = BaggingRegressor(TreePruned)

# params = [{
#     "n_estimators": np.arange(10, 100, 10),
#     'max_samples': np.arange(0.01, 1, 0.01),
# }]

# bag_search = GCV(BagginTree,
#                  param_grid=params,
#                  cv=5,
#                  verbose=0,
#                  return_train_score=True)

# bag_search.fit(x_train, y_train)

# best_params = bag_search.best_params_
# best_estimator = bag_search.best_estimator_

best_estimator = BaggingTree.fit(x_train, y_train)

print("--------Inciso F - Bagging--------")
print("Score sobre train: {}".format(best_estimator.score(x_train, y_train)))
print("Score sobre test: {}".format(best_estimator.score(x_test, y_test)))

features_importance = np.array([
    estimator.feature_importances_ for estimator in best_estimator.estimators_
]).mean(axis=0)

print('Importancia de parámetros: {}'.format(-np.sort(-features_importance)))
print('Orden por importancia descendente: {}'.format(
    np.array(x_train.keys()[np.argsort(-features_importance)])))

#### Inciso G #####
print("-------- Inciso G - Random Forest --------")

from sklearn.ensemble import RandomForestRegressor

RForest = RandomForestRegressor()

# params = [{
#     "n_estimators": np.arange(10, 100, 10),
# }]

# forest_search = GCV(RForest,
#                  param_grid=params,
#                  cv=5,
#                  verbose=0,
#                  return_train_score=True)

# forest_search.fit(x_train, y_train)

# best_params = forest_search.best_params_
# best_estimator = forest_search.best_estimator_

RForest.fit(x_train, y_train)
RForest.score(x_test, y_test)

features_importance = np.array([
    estimator.feature_importances_ for estimator in RForest.estimators_
]).mean(axis=0)

score_train = []
score_test = []
depths = np.arange(1, 30)

for depth in depths:
    RForest = RandomForestRegressor(max_depth=depth)
    RForest.fit(x_train, y_train)
    score_train.append(RForest.score(x_train, y_train))
    score_test.append(RForest.score(x_test, y_test))

print('Score sobre test: {}'.format(np.max(score_test)))

figsize = (6, 5)
fs = 16
lw = 3

test_max = np.argmax(score_test)

plt.figure(figsize=figsize)
plt.plot(depths, score_train, label='Train', lw=lw, marker='o')
plt.plot(depths, score_test, label='Test', lw=lw, marker='o')
plt.xlabel('Profundidad', fontsize=fs)
plt.ylabel('Score', fontsize=fs)
plt.title(r'Random Forest - $max features$ libre', fontsize=fs)
plt.legend(fontsize=fs)
plt.tick_params(labelsize=fs)
# plt.annotate('Maximo para testing', xy=(test_max, score_test[test_max]))
plt.tight_layout()
plt.savefig('RForest_depth.pdf')
plt.show()

score_train = []
score_test = []
max_features = np.arange(1, x_train.shape[1])

for feature in max_features:
    RForest = RandomForestRegressor(max_features=feature)
    RForest.fit(x_train, y_train)
    score_train.append(RForest.score(x_train, y_train))
    score_test.append(RForest.score(x_test, y_test))

print('Score sobre test: {}'.format(np.max(score_test)))

fs = 16
lw = 3

plt.figure(figsize=figsize)
plt.plot(max_features, score_train, label='Train', lw=lw, marker='o')
plt.plot(max_features, score_test, label='Test', lw=lw, marker='o')
plt.xlabel('Numero de caracteristicas', fontsize=fs)
plt.ylabel('Score', fontsize=fs)
plt.title(r'Random Forest - $max depth$ libre', fontsize=fs)
plt.legend(fontsize=fs)
plt.tick_params(labelsize=fs)
# plt.annotate('Maximo para testing',
#              arrowstyle='->',
#              xy=(test_max, score_test[test_max]))
plt.tight_layout()
plt.savefig('RForest_features.pdf')
plt.show()

print('Importancia de parámetros: {}'.format(-np.sort(-features_importance)))
print('Orden por importancia descendente: {}'.format(
    np.array(x_train.keys()[np.argsort(-features_importance)])))

#### Incido H ####
print("--------Inciso H - AdaBoost --------")
from sklearn.ensemble import AdaBoostRegressor

ABoost = AdaBoostRegressor(base_estimator=DecisionTreeRegressor())
ABoost.fit(x_train, y_train)
ABoost.score(x_test, y_test)

features_importance = np.array([
    estimator.feature_importances_ for estimator in ABoost.estimators_
]).mean(axis=0)

print('Importancia de parámetros: {}'.format(-np.sort(-features_importance)))
print('Orden por importancia descendente: {}'.format(
    np.array(x_train.keys()[np.argsort(-features_importance)])))

score_train = []
score_test = []
depths = np.arange(1, 30)

for depth in depths:
    ABoost = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(
        max_depth=depth))
    ABoost.fit(x_train, y_train)
    score_train.append(ABoost.score(x_train, y_train))
    score_test.append(ABoost.score(x_test, y_test))

print('Score sobre test: {}'.format(np.max(score_test)))

figsize = (6, 5)
fs = 16
lw = 3

test_max = np.argmax(score_test)

plt.figure(figsize=figsize)
plt.plot(depths, score_train, label='Train', lw=lw, marker='o')
plt.plot(depths, score_test, label='Test', lw=lw, marker='o')
plt.xlabel('Profundidad', fontsize=fs)
plt.ylabel('Score', fontsize=fs)
plt.title(r'AdaBoost - $max features$ libre', fontsize=fs)
plt.legend(fontsize=fs)
plt.tick_params(labelsize=fs)
# plt.annotate('Maximo para testing',
#              arrowstyle='->',
#              xy=(test_max, score_test[test_max]))
plt.tight_layout()
plt.savefig('Adaboost_depth.pdf')
plt.show()

score_train = []
score_test = []
max_features = np.arange(1, x_train.shape[1])

for feature in max_features:
    ABoost = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(
        max_features=feature))
    ABoost.fit(x_train, y_train)
    score_train.append(ABoost.score(x_train, y_train))
    score_test.append(ABoost.score(x_test, y_test))

print('Score sobre test: {}'.format(np.max(score_test)))

fs = 16
lw = 3

plt.figure(figsize=figsize)
plt.plot(max_features, score_train, label='Train', lw=lw, marker='o')
plt.plot(max_features, score_test, label='Test', lw=lw, marker='o')
plt.xlabel('Numero de caracteristicas', fontsize=fs)
plt.ylabel('Score', fontsize=fs)
plt.title(r'Adaboost - $max depth$ libre', fontsize=fs)
plt.legend(fontsize=fs)
plt.tick_params(labelsize=fs)
# plt.annotate('Maximo para testing',
#              arrowstyle='->',
#              xy=(test_max, score_test[test_max]))
plt.tight_layout()
plt.savefig('Adaboost_features.pdf')
plt.show()
