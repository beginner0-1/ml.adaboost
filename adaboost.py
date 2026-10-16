from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import DecisionTreeClassifier

x ,y = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=10,
    n_redundant=2,
    random_state=42
)
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=42)

from sklearn.ensemble import AdaBoostClassifier

base_model = DecisionTreeClassifier(
    max_depth=1
)

abc = AdaBoostClassifier(
    estimator=base_model,
    n_estimators=100,
    random_state=42

)

abc.fit(x_train,y_train)

y_pred= abc.predict(x_test)

print("accuracy:", accuracy_score(y_test,y_pred))
print("classification report ", classification_report(y_test,y_pred))
