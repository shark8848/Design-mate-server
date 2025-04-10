import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class PolyRegressor:
    def __init__(self):
        
        self.params = None
        # 以下数据: x 为实际的各类窗户面积,y 为对应的窗框与窗户面积比
        self.x = np.array([
                            0.9899, 1.0384, 1.2147, 1.0684, 1.11, 1.1932, 1.3198, 1.6135, 
                            0.987, 1.0256, 1.1028, 0.9824, 1.1182, 1.1993, 1.321, 1.2951, 
                            1.7734, 1.9791, 2.185, 2.1314, 4.6088, 2.341, 1.7126, 1.7775, 
                            2.2301, 2.8479, 3.5546, 3.8684, 3.2519, 2.7309, 2.9728, 3.2403, 
                            3.4256, 3.1486, 3.5388, 3.5644, 3.669, 3.8784, 4.1923, 4.4444,
                             4.1895, 4.5575, 4.3979, 4.5266, 4.7839, 5.1699, 4.6063, 4.8147, 
                             4.9553, 5.2366, 5.6586, 5.463, 4.7358, 5.0023, 2.7333, 3.0416, 
                             8.0827, 8.5393, 9.0828, 9.8734, 13.5695, 14.5356, 14.5983, 15.6364, 
                             15.1127, 15.4707, 16.1868, 16.6559, 17.838, 18.206, 19.0436, 18.3379, 
                             20.1383, 20.6118, 20.6527])
        self.y = np.array([
                            0.440650571, 0.434322034, 0.377871079, 0.422688132, 0.411351351, 
                            0.390965471, 0.366040309, 0.320545398, 0.45633232, 0.444032761, 
                            0.422016685, 0.482390065, 0.371221606, 0.363295256, 0.35321726, 
                            0.381592155, 0.337712868, 0.310090445, 0.287688787, 0.296847143, 
                            0.178245964, 0.232806493, 0.527093308, 0.521012658, 0.492892695, 
                            0.41374346, 0.228548923, 0.213886878, 0.228297303, 0.258925629, 
                            0.242599569, 0.228373916, 0.218764596, 0.233849965, 0.21377303, 
                            0.214201549, 0.209375852, 0.20049505, 0.188846218, 0.182904329, 
                            0.192337988, 0.179901262, 0.186429887, 0.182167631, 0.174334748, 
                            0.164045726, 0.181056379, 0.176148047, 0.172098561, 0.164648818, 
                            0.154861627, 0.16040637, 0.185480806, 0.177598305, 0.254454323, 
                            0.242503945, 0.186373365, 0.180248967, 0.185372352, 0.176666599, 
                            0.214768414, 0.206231597, 0.202372879, 0.19427106, 0.196807983, 
                            0.194050689, 0.188900833, 0.182175685, 0.174778563, 0.141997144, 
                            0.138272175, 0.130876491, 0.122651862, 0.120814291, 0.12056535
                        ])
        self.fit(self.x, self.y)

    def log_func(self, x, a, b):
        return a * np.log(x) + b

    def fit(self, x, y):
        self.params, _ = curve_fit(self.log_func, x, y, p0=[0.5, 0.5])
        
    def predict(self, x):
        if self.params is None:
            raise Exception("Model is not fitted yet. Call 'fit' with appropriate data.")
        return self.log_func(x, self.params[0], self.params[1])

    def plot(self, start=1, end=25):
        plt.figure(figsize=(10, 6))
        # scatter plot of the original data
        plt.scatter(self.x, self.y, color='blue', label='Original data')

        # plot of the fitted curve
        x_range = np.linspace(start, end, 400)
        y_range = self.predict(x_range)
        plt.plot(x_range, y_range, color='red', label='Fitted curve')

        # labels, title and legend
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Curve fitting')
        plt.legend()

        # show the plot
        plt.show()

if __name__ == "__main__":
    arp = PolyRegressor()
    arp.plot()
    for i in range(10, 25):
        prediction = arp.predict(i)
        print("x=",i, "y=", round(prediction,4), "z=", round(i*prediction,4))
