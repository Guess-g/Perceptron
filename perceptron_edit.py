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
def train_and_gate():
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

#TRAINING OR GATE
def train_or_gate():
    x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train_or = [0, 1, 1, 1]

    or_brain = Perceptron(2, 0.1)
    print("Training OR gate...")
    or_brain.train(x_train, y_train_or, 20)

    print("\nFinal Perceptron Stats")
    print(f"Weights: {or_brain.weights}")
    print(f"Bias: {or_brain.bias}\n")

    print("Testing OR gate...")
    for inputs in x_train: 
        prediction = or_brain.predict(inputs)
        print(f"Input: {inputs}, Prediction: {prediction}")

#TRAINING NAND GATE
def train_nand_gate():
    x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train_nand = [1, 1, 1, 0]

    nand_brain = Perceptron(2, 0.1)
    print("Training NAND gate...")
    nand_brain.train(x_train, y_train_nand, 20)

    print("\nFinal Perceptron Stats")
    print(f"Weights: {nand_brain.weights}")
    print(f"Bias: {nand_brain.bias}\n")

    print("Testing NAND gate...")
    for inputs in x_train: 
        prediction = nand_brain.predict(inputs)
        print(f"Input: {inputs}, Prediction: {prediction}")

#TRAINING NOR GATE
def train_nor_gate():
    x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train_nor = [1, 0, 0, 0]

    nor_brain = Perceptron(2, 0.1)
    print("Training NOR gate...")
    nor_brain.train(x_train, y_train_nor, 20)

    print("\nFinal Perceptron Stats")
    print(f"Weights: {nor_brain.weights}")
    print(f"Bias: {nor_brain.bias}\n")

    print("Testing NOR gate...")
    for inputs in x_train: 
        prediction = nor_brain.predict(inputs)
        print(f"Input: {inputs}, Prediction: {prediction}")

#MENU
while True:
    print("\nSelect a gate to train:")
    print("1. AND")
    print("2. OR")
    print("3. NAND")
    print("4. NOR")

    choice = input("Enter the number of the gate you want to train: ")

    if choice == "1":
        train_and_gate()
    elif choice == "2":
        train_or_gate()
    elif choice == "3":
        train_nand_gate()
    elif choice == "4":
        train_nor_gate()
    else:
        print("Invalid choice. Please select a valid gate number.")
