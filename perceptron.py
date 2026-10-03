import random

#CREATION OF PERCEPTRON
class Perceptron:
    def __init__(self, num_weights, learning_rate=0.1): #for intisialiasing any new ai (perceptron)
        self.weights = []
        for i in range(num_weights):
            self.weights.append(random.uniform(-1, 1)) #u use 'uniform' instead of 'randint' so floats are stored. if we used integers, changes would be too big and itd never learn
        self.bias = 0.0
        self.lr = learning_rate #keep in mind 0.1 is a standard lr for small AI problems. 
        print(f"Initial weights: {self.weights}")
        print(f"Initial bias: {self.bias}\n")

    def activation(self, x):
        if x > 0:
            return 1
        else:
            return 0

    def predict(self, inputs):
        weighted_sum = 0
        for i, w in zip(inputs, self.weights):
            weighted_sum += i * w
        weighted_sum += self.bias  
        return self.activation(weighted_sum)

    def train(self, training_data, answers, rounds=20):
        for _ in range(rounds): #outer loop for how many times it is acc trained
            total_errors = 0
            for inputs, ans in zip(training_data, answers): #inner loop passes through acc training data
                prediction = self.predict(inputs)
                error = ans - prediction

                if error != 0:
                    total_errors += abs(error)
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * inputs[i]
                    self.bias += self.lr * error 

            print(f"Round: {_ + 1}")
            print(f"Total errors: {total_errors}\n")

            if total_errors == 0:
                print("Training completed early.")
                break


#TRAINING AND GATE
x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train_and = [0, 0, 0, 1]

and_brain = Perceptron(2, 0.1)
print("Training AND gate...")
and_brain.train(x_train, y_train_and, 20)

print("\nFinal Perceptron Stats")
print(f"Weights: {and_brain.weights}")
print(f"Bias: {and_brain.bias}\n")

print("Testing AND gate...")
for inputs in x_train: 
    prediction = and_brain.predict(inputs)
    print(f"Input: {inputs}, Prediction: {prediction}")