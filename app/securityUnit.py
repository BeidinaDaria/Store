import pandas as pd
from numpy import random, append,array
from math import log2, sqrt
from collections import Counter
import json, os

class KNN:
    def __init__(self, X, Y):
        self.X_train = array(X.values)
        self.Y_train = array(Y.values)
        self.k = Counter(self.Y_train).most_common(1)[0][1] // 10

    def predict(self, x):
        """Предсказание класса для новых данных."""
        neighbors = self._get_neighbors(x)
        most_common = Counter(neighbors).most_common(1)
        prediction=most_common[0][0]
        self.X_train=self.X_train[1:]
        self.Y_train=self.Y_train[1:]
        self.X_train = append(self.X_train, x)
        self.Y_train = append(self.Y_train, most_common[0][0])
        return prediction

    def _euclidean_distance(self, a, b):
        """
        Евклидово расстояние без извлечения квадратного корня,
        что позволяет ускорить вычисление.
        """
        s=0
        for i in range(0,len(a)):
            s+=(a[i]- b[i]) ** 2
        return sqrt(s)
    
    def _get_neighbors(self, x):
        """Возвращает k ближайших соседей для заданной точки x"""
        distances = []
        for i in range(0, len(self.X_train)):
            distances.append((self.Y_train[i], self._euclidean_distance(x, self.X_train[i])))
        distances.sort(key=lambda item: item[1])  
        nearest_neighbors = [item[0] for item in distances[:self.k]]
        return nearest_neighbors

class DecisionTreeClassifier:
    def __init__(self, X_train, Y_train, depth):
        self.max_depth = depth
        self.min_samples_leaf = 100
        self.tree=self.build_tree(X_train,Y_train)

    # Вычисление энтропии множества меток y
    @staticmethod
    def entropy(y):
        counts = Counter(y)
        probabilities = [count / len(y) for count in counts.values()]
        return -sum(p * log2(p) for p in probabilities if p != 0)

    
    # Поиск оптимального порога разбиения по признаку feature_idx
    def find_best_split(self, X, y, feature_idx):
        column_values = X[X.columns[feature_idx]].sort_values().values
        thresholds = [(column_values[i] + column_values[i+1])/2 for i in range(len(column_values)-1)]
        
        best_entropy = float('inf')
        best_threshold = None
        
        for threshold in thresholds:
            left_mask = X[X.columns[feature_idx]] <= threshold
            right_mask = ~left_mask
            
            left_y = y[left_mask]
            right_y = y[right_mask]
            
            if len(left_y) == 0 or len(right_y) == 0:
                continue
            
            total_samples = len(y)
            left_entropy = self.entropy(left_y)
            right_entropy = self.entropy(right_y)
            
            weighted_entropy = (len(left_y)/total_samples)*left_entropy + (len(right_y)/total_samples)*right_entropy
            
            if weighted_entropy < best_entropy:
                best_entropy = weighted_entropy
                best_threshold = threshold
                
        return best_threshold, best_entropy

    def build_tree(self, X, y, depth=0):
        if (depth >= self.max_depth or 
            len(set(y)) == 1 or 
            len(X) <= self.min_samples_leaf):
            return {
                'leaf': True,
                'value': Counter(y).most_common(1)[0][0]
            }
        
        best_feature = None
        best_threshold = None
        min_entropy = float('inf')
        
        for feature_idx in range(len(X.columns)):
            threshold, entropy = self.find_best_split(X, y, feature_idx)
            if entropy < min_entropy:
                min_entropy = entropy
                best_feature = feature_idx
                best_threshold = threshold
        
        node = {}
        node['feature'] = X.columns[best_feature]
        node['threshold'] = best_threshold
        
        left_mask = X[X.columns[best_feature]] <= best_threshold
        right_mask = ~left_mask
        
        node['left'] = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        node['right'] = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        print(node['feature'], node['threshold'])
        return node
    
    def predict(self, X):
        # """
        # Предсказывает класс для входных данных X
        
        # :param X: матрица признаков (numpy array или pandas DataFrame)
        # :return: список предсказанных классов
        # """ 
        predictions = []
        for index,sample in X.iterrows():
            predictions.append(self._predict_sample(sample, self.tree))
        return predictions
    
    def _predict_sample(self, sample, node):
        # """
        # Вспомогательная рекурсивная функция для предсказания одного образца       
        # :param sample: вектор признаков одного образца
        # :param node: текущий узел дерева
        # :return: предсказанный класс
        # """
        # Если достигли листа, возвращаем его значение
        
        if 'leaf' in node:
            return node['value']
        # Переход влево или вправо зависит от значения признака
        if sample[node['feature']] <= node['threshold']:
            return self._predict_sample(sample, node['left'])
        else:
            return self._predict_sample(sample, node['right'])

class RandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=None, random_state=None):
        # """
        # Параметры:
        # n_estimators: количество деревьев
        # max_depth: максимальная глубина деревьев
        # max_features: число признаков для рассмотрения при каждом разделении ('sqrt', 'log2', или int)
        # random_state: фиксирует случайность
        # """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
    
    def fit(self, X, y):
        # """Обучение случайного леса"""   
        n_samples, n_features = X.shape

        # Обучаем каждое дерево
        for _ in range(self.n_estimators):
            # Bootstrap выборка
            sample_indices = random.choice(n_samples, n_samples//n_features*2, replace=False)
            n_features_per_tree=random.randint(n_features//2+1, n_features)
            # Выбираем случайные признаки
            feature_indices = random.choice(
                n_features, 
                n_features_per_tree, 
                replace=False
            )
            features=[]
            for i in feature_indices:
                features.append(X.columns[i])
            X_subset = X.loc[sample_indices, features]
            y_sample=y[sample_indices]
            
            # Создаем и обучаем дерево
            tree = DecisionTreeClassifier(X_subset, y_sample,self.max_depth)
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        # """Предсказание класса для входных данных"""
            
        # Получаем предсказания от всех деревьев
        all_predictions = []
        for tree, feature_indices in self.trees:
            features=[]
            for i in feature_indices:
                features.append(X.columns[i])
            X_subset = X.loc[:, features]
            all_predictions.append(tree.predict(X_subset))
        
        # Голосование большинством
        predictions = []
        for i in range(X.shape[0]):
            preds = [pred[i] for pred in all_predictions]
            predictions.append(Counter(preds).most_common(1)[0][0])
            
        return predictions

class IntrusionDetectionSystem:
    def __init__(self):
        # Загрузка данных для обучения
        self.knn_train = pd.read_csv('knn_dataset.csv')
        self.knn_res = self.knn_train['class']
        self.knn_train=self.knn_train.drop('class', axis=1)
        self.rf_train = pd.read_csv('rf_dataset.csv')
        self.rf_res = self.rf_train['class']
        self.rf_train=self.rf_train.drop('class', axis=1)

        # Инициализация и обучение моделей
        self.knn = KNN(self.knn_train, self.knn_res)
        self.rf = RandomForestClassifier(n_estimators=15, max_depth=5)
        self.rf.fit(self.rf_train, self.rf_res)
    
    def check_ip_in_json(self, ip_address):
        """Проверка IP-адреса в JSON-файле."""
        if not os.path.exists('ipBase.json'):
            return False

        with open('ipBase.json', 'r') as file:
            data = json.load(file)

        ips_set = set(data.get('ips', []))
        return ip_address in ips_set
    
    def check(self, ip_address, features):
        """
        Проверка IP-адреса на безопасность.
        
        :param ip_address: IP-адрес для проверки
        :param features: словарь с признаками
        :return: результат проверки (True/False или тип атаки)
        """
        # 1. Проверка по белому списку
        if self.check_ip_in_json(ip_address):
            print('blocked ip')
            return True
        
        # 3. Проверка через KNN (аномалии)
        knn_pred = self.knn.predict(array(list(features.values())))[0]
        if knn_pred == 'normal':
            print('normal package')
            return False
        
        # 4. Проверка через RandomForest (точная классификация)
        rf_pred = self.rf.predict(pd.DataFrame([features], columns=features.keys()))[0]
        print(rf_pred)
        if (rf_pred=='normal'):
            return False
        return rf_pred