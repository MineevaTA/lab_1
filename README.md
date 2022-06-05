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

$$ 1/(2*n) sum_{i=1}^{n} (y_(pred)-y)^2 = 1$
        
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












