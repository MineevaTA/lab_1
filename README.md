# lab_1

В данной работе реализованы различные алгоритмы для классификации для решения задачи определения гендера человека по некоторым признакам. За основу взят тот же датасет, что использовался в лабораторной работе №0 - Gender Classification Dataset.

# Линейная регрессия

Задаем два гиперпараметра - количество эпох и learning rate.

    def __init__(self, epochs, lr):
          self.lr = lr
          self.epochs = epochs


Реализуем формулы линейной регрессии. Вычисляем предсказания как:

$$Y = X*w+b$$

    def predict(self, X):
          return X.dot(self.w)+self.b
        
Функция потерь:

$$ 1/(2*n) \sum_{i=1}^{n} (y_(pred)-y)^2$$
        
    def loss(self, y_pred, y):
          return np.sum(np.square(y_pred-y))/(2*self.N)
      
Далее пишем функцию fit.

    def fit(self, X, y):
          self.w = np.zeros(X.shape[1])
          self.b = 0
          self.N = X.shape[0]   
        
          for i in range(self.epochs):
              y_pred = np.dot(X,self.w) + self.b
              dw = np.dot(X.T,(y_pred-y)) / self.N
              db = np.sum(y - y_pred)  / self.N
              self.w = self.w - self.lr*dw
              self.b = self.b - self.lr*db
            
Здесь мы задаем параметры модели w = 0, b = 0, N - длина входного массива. Запускаем обучение по эпохам. За каждую эпоху мы считаем предсказания y_pred, по ним находим градиенты функции. Шагаем по градиентам, получая новые параметры w и b 

С помощью GridSearchCV вычисляем оптимальные гиперпараметры модели. 

    parameters_grid = {
      'lin__epochs': [10, 15, 20, 50, 100],
      'lin__lr': [0.001, 0.01, 0.1, 1],
    }

    grid_cv = GridSearchCV(pipe, parameters_grid,scoring = 'neg_mean_squared_error')
    grid_cv.fit(X_train, y_train)
  
Получаем epochs = 100, lr = 0.01
  
Используем основные метрики для оценивания точности модели. Получаем:
  
    MSE:  0.2121535002004107
    MAE:  0.4592512403577702
    RMS:  0.46060123773217404
    
А также выводим матрицу неточностей:

    array([[403,  77],
         [ 66, 455]])
       
Сравниваем оценки с sklearn.linear_model.LinearRegression:
 
    MSE:  0.03994022299855651
    MAE:  0.14737544554039292
    RMS:  0.1998505016219787 
  
    array([[465,  15],
         [ 22, 499]])
       
Видно,что у моей модели невысокая точность. Опытным путем было обнаружено, что это частично исправляется добавлением количества эпох - в таком случае точность повышается, но достаточно медленно.

# Метод опорных векторов

Задаем те же гиперпараметры, что и в предыдущем случае:

    def __init__(self, lr=1, epochs=1000):
        self.lr = lr        
        self.epochs = epochs
        
 Предсказания вычисляются по формуле модели:
 
 $$ class(x) = sign(X*w)$$
 
    def predict(self, X):
        y_pred = np.dot(X, self.w)
        return np.where(y_pred > 0, 1, 0)
        
Далее пишем функцию fit.

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        y_ = np.where(y > 0, 1, -1)

        for e in range(self.epochs):
            for i, x in enumerate(X):
                class_x = y_[i] * (np.dot(x, self.w))
                if class_x < 1:
                    self.w += self.lr * (X[i]*y_[i] - 2/self.epochs*self.w)
                else:
                    self.w += self.lr * (-2/self.epochs*self.w)
                    
Здесь мы задаем параметр w = 0, строим вектор y_ -  преобразованными к нужному нам виду(два класса) вектор y. Для каждого x вычисляем функцию потерь. В зависимости от ее значения меняем параметр w.

С помощью GridSearchCV вычисляем оптимальные гиперпараметры модели. 

        parameters_grid = {
        'svm__lr': [0.001, 0.01, 0.1],    
        'svm__epochs': [20, 50]    
        }

        grid_cv = GridSearchCV(pipe, parameters_grid)
        grid_cv.fit(X_train, y_train)
  
Получаем epochs = 20, lr = 0.001.

Используем основные метрики для оценивания точности модели. Получаем:
  
    MSE:  0.04195804195804196
    MAE:  0.04195804195804196
    RMS:  0.20483662259967567
    Roc_auc:   0.9566598688419705
    
А также выводим матрицу неточностей:

    array([[443,  37],
        [  5, 516]])
       
Сравниваем оценки с sklearn.svm.SVC:
 
    MSE:  0.03196803196803197
    MAE:  0.03196803196803197
    RMS:  0.17879606250706967
    Roc_auc:   0.9682241682661549

  
    array([[467,  13],
        [ 19, 502]])

Моя реализация практически догоняет по точности sklearn.svm.SVC.

# Метод ближайших соседей 

В модели один гиперпараметр - количество ближайших соседей.

    def __init__(self, k = 4):
        self.k = k
        
Функция fit служит для передачи массивов X и y - "соседей":

    def fit(self, X, y):
        self.X = X
        self.y = y
        
 Далее рассмотрим функцию predict:
 
    def predict(self, x):
        y_pred = []
    
        for elem in x: 
            distances = []
            for j in range(len(self.X)): 
                distances.append(np.sum(np.absolute(np.array(self.X[j,:]) - nbr))) 
            
            distances = np.array(distances) 
            nearest_x = np.argsort(distances)[:self.k] 
            nearest_y = self.y[nearest_x]

            ans = mode(nearest_y)
            ans = ans.mode[0]
            y_pred.append(ans)

        return y_pred
        
Здесь мы создаем пустой массив для предсказаний модели. Далее для каждого элемента из x мы вичисляем расстояния между этим элементом и элементами из X, помещая полученные значения в массив distances. Сортируем массив для получения именно "ближайших" соседей, берем только k из них. Для каждого полученного элемента определяем его класс, записываем в массив. С помощью функции mode() определяем наиболее часто встречающийся класс из полученного массива - именно он становится классом элемента x.


С помощью GridSearchCV вычисляем оптимальное количество соседей. 

    parameters_grid = {
        'knn__k': [2, 5, 7, 10]  
    }

        grid_cv = GridSearchCV(pipe, parameters_grid)
        grid_cv.fit(X_train, y_train)
        
Получаем k = 7.

Оценки точности модели:

    MSE:  0.03196803196803197
    MAE:  0.03196803196803197
    RMS:  0.17879606250706967
    Roc_auc:   0.9682241682661549

    array([[467,  13],
       [ 19, 502]])

Сравнение с KNeighborsClassifier:

      MSE:  0.028971028971028972
      MAE:  0.028971028971028972
      RMS:  0.17020878053446295
      Roc_auc:   0.971431142034549

      array([[471,   9],
             [ 20, 501]])

Модель так же приближается по точности к KNeighborsClassifier, даже показывает большую точность при использовании оченки RMS.  






