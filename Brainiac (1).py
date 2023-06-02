import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import openai
openai.api_key = "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"

from openai.api_resources import Completion
import sounddevice as sd
import speech_recognition as sr


## TODO: modules sd, sr missing. sound recording not possible.

import sys
openai.api_key = "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"
import openai
from openai.api_resources import Completion
import re
import time

openai.api_key = "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"

############################################################################################################### CLASS AUDIO RECORDER##############################################################################################

class AudioRecorder:
    openai.api_key = "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"

    def __init__(self, freq=44100, duration=5):
        self.freq = freq
        self.duration = duration


    def record(self):
        recording = sd.rec(int(self.freq * self.duration), samplerate=self.freq, channels=1)
        sd.wait()  # Wait until recording is finished
        return np.squeeze(recording)


### TODO: this function does nothing, it is supposed to check for physical neurons to attach to
def get_sensor_data():
    return 10*[1,]

### TODO: this function does nothing, it is for neuron integration
def preprocess_data(data):
    return data

def compute_error(a, expected_input):
    return a



## Sigmoid functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def sigmoid_prime(x):
    return self.sigmoid_derivative(x)
###



class NeuralNet:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        # Initialize the weights
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.learning_rate = learning_rate

    def forward(self, X, sensor_data):
        # Preprocess the sensor data
        processed_data = preprocess_data(sensor_data)

        # Concatenate the sensor data with the input data
        X = np.concatenate((X, processed_data), axis=1)

        # First layer
        Z1 = np.dot(X, self.W1)
        A1 = sigmoid(Z1)

        # Second layer
        Z2 = np.dot(A1, self.W2)
        A2 = sigmoid(Z2)

        return A2

    def backward(self, X, y, sensor_data):
        # Preprocess the sensor data
        processed_data = preprocess_data(sensor_data)

        # Concatenate the sensor data with the input data
        X = np.concatenate((X, processed_data), axis=1)

        # Forward pass
        A2 = self.forward(X, sensor_data)

        # Compute the error
        error = compute_error(A2, y)

        # Backpropagation
        dZ2 = A2 - y
        dW2 = np.dot(A1.T, dZ2)
        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_prime(A1)
        dW1 = np.dot(X.T, dZ1)

        # Update the weights
        self.W1 -= self.learning_rate * dW1
        self.W2 -= self.learning_rate * dW2

        return error

class NeuralNetwork:
# Define the neural network architecture
    input_size = 10 # number of input neurons
    hidden_size = 5 # number of hidden neurons
    output_size = 1 # number of output neurons
    learning_rate = 0.1 # learning rate

# Initialize the weights
    W1 = np.random.randn(input_size, hidden_size)
    W2 = np.random.randn(hidden_size, output_size)

####### from NeuralNet:
    #def __init__(self, input_size, hidden_size, output_size, learning_rate):
    def init2(self):
        # Initialize the weights
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.learning_rate = learning_rate

    def forward(self, X, sensor_data):
        # Preprocess the sensor data
        processed_data = preprocess_data(sensor_data)

        # Concatenate the sensor data with the input data
        X = np.concatenate((X, processed_data), axis=1)

        # First layer
        Z1 = np.dot(X, self.W1)
        A1 = sigmoid(Z1)

        # Second layer
        Z2 = np.dot(A1, self.W2)
        A2 = sigmoid(Z2)

        return A2

    def backward(self, X, y, sensor_data):
        # Preprocess the sensor data
        processed_data = preprocess_data(sensor_data)

        # Concatenate the sensor data with the input data
        X = np.concatenate((X, processed_data), axis=1)

        # Forward pass
        A2 = self.forward(X, sensor_data)

        # Compute the error
        error = compute_error(A2, y)

        # Backpropagation
        dZ2 = A2 - y
        dW2 = np.dot(A1.T, dZ2)
        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_prime(A1)
        dW1 = np.dot(X.T, dZ1)

        # Update the weights
        self.W1 -= self.learning_rate * dW1
        self.W2 -= self.learning_rate * dW2

        return error
######
    def backward2(self):
    # Collect data from sensors
        sensor_data = get_sensor_data()

    # Preprocess the data
        processed_data = preprocess_data(sensor_data)

        # Feed the data through the neural network
        # First layer
        Z1 = np.dot(processed_data, self.W1)
        A1 = sigmoid(Z1)

        # Second layer
        Z2 = np.dot(A1, W2)
        A2 = sigmoid(Z2)

        # Compute the error
        error = compute_error(A2, expected_output)

        # Backpropagate the error
        dZ2 = A2 - expected_output
        dW2 = np.dot(A1.T, dZ2)
        dZ1 = np.dot(dZ2, W2.T) * sigmoid_prime(A1)
        dW1 = np.dot(processed_data.T, dZ1)

    def __init__(self, input_size, output_size, num_hidden):
        self.num_inputs = input_size
        self.output_size = output_size
        self.hidden_layer = np.zeros(num_hidden)
        self.output_layer = np.zeros(self.output_size)
        self.W1 = np.random.rand(self.num_inputs, num_hidden)
        self.W2 = np.random.rand(num_hidden, self.output_size)
        self.trainable_variables = [self.W1, self.W2]


    def activate(self, inputs):
        self.hidden_layer = np.dot(inputs, self.weights1)
        self.hidden_layer = sigmoid(self.hidden_layer)
        self.output_layer = np.dot(self.hidden_layer, self.W2)
        self.output_layer = sigmoid(self.output_layer)
        return self.output_layer

    def backpropagate(self, inputs, y, learning_rate):
        output_error = y - self.output_layer
        output_delta = output_error * sigmoid_derivative(self.output_layer)
        hidden_error = np.dot(output_delta, self.W2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer)
        self.W2 += learning_rate * np.dot(self.hidden_layer.T, output_delta)
        self.W1 += learning_rate * np.dot(inputs.T, hidden_delta)


    def train(self, inputs, targets, num_epochs, learning_rate):
        for i in range(num_epochs):
            for j in range(len(inputs)):
                self.activate(inputs[j])
                self.backpropagate(inputs, targets[j], learning_rate)
openai.api_key = "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"


class Body: # class defined later
    pass

class Environment: #class defined later
    pass
openai.api_key = "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"

class AI:
    def __init__(self):
        text_model_data = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.text_model = hub.KerasLayer(text_model_data)
        #image_model_data = hub.load("https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/5")
        #self.image_model = hub.KerasLayer(image_model_data)
        self.num_neurons = 10
        self.network = NeuralNetwork(input_size=3+self.num_neurons, output_size=1, num_hidden=self.num_neurons)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.memory = ""
        ##TODO: output shape cannot be determined before the model is called
        #self.attention = np.zeros(self.text_model.output_shape[0][-1])
        self.attention = np.zeros(self._get_output_size(), dtype=np.float32)
        self.emotion = np.zeros(3) # 3-dimensional vector representing positive, negative, and neutral emotions
        self.sensor_data = []
        self.body_actions = []
        self.neurons = []
        self.env = Body()
    def __init__(self):
        self.completion = openai.Completion()

# Authenticate to the OpenAI API using your API key
openai.api_key = "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"

# Load the text model
text_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Define the Brainiac class
class Brainiac:
    def __init__(self, text_model, max_tokens=1024):
        self.text_model = text_model
        self.max_tokens = max_tokens

    def generate_response(self, prompt):
        # Generate a response using OpenAI's GPT-3 engine
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=self.max_tokens
        )["choices"][0]["text"]

        # Remove any unnecessary whitespace from the response
        response = re.sub(r"\n\s*\n", "\n", response.strip())

        # Return the response
        return response

# Define the main function
def main():
    # Create an instance of the Brainiac class
    ai = Brainiac(text_model)

    print("Hello, I'm Brainiac! How can I assist you today?")

    while True:
        prompt = input("P: ")
        if prompt.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            print("Brainiac: Goodbye!")
            break
        else:
            response = ai.generate_response(prompt)
            print("Brainiac:", response)

if __name__ == "__main__":
    main()
# Define logic flow prompt for ai multipoint responses
prompt = (
    f"Please provide me with your name and contact details.\n"
    f"\n"
    f"Name: John Doe\n"
    f"Phone: 555-1234\n"
    f"Email: john.doe@example.com\n"
    f"\n"
    f"I am interested in hosting an event at your facility.\n"
    f"\n"
    f"Event: Wedding reception\n"
    f"Date: August 1st, 2023\n"
    f"Number of guests: 150\n"
    f"\n"
    f"Additional details: I am looking for a venue that can accommodate both the wedding ceremony and the reception. "
    f"I would also like to have access to outdoor space for photos and a cocktail hour. "
    f"Do you have in-house catering or should I hire an outside caterer? "
    f"I'm interested in a sit-down dinner, can your facility accommodate that? "
    f"I would also like to have a band, is there a stage or suitable place for them? "
    f"Do you provide any wedding planning services or coordination on the day of the event? "
    f"I am also interested in the audio-visual equipment you provide, could you give me more information on that? "
    f"Is there sufficient parking space for guests, and is it included in the price? "
    f"Can you provide me with more information about your facility and services?"
    f"\n"
)
# Create an instance of the Brainiac class
ai = Brainiac(text_model)

# Generate a response
response = ai.generate_response(prompt)
print(response)



###############Removed the annoying text recall errors############################


class AI2:
    def __init__(self):
        self.text_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.num_neurons = 10
        self.network = NeuralNetwork(input_size=3+self.num_neurons, output_size=1, num_hidden=self.num_neurons)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


    def generate_prompt(self, prompt):
        prompt += " " + self.memory
        attention = tf.nn.softmax(tf.keras.layers.Dense(self.text_model.get_output_shape_at(0)[-1])(self.text_model(prompt) * tf.expand_dims(self.attention, axis=0)))
        prompt += " " + self.add_emotion(self.emotion)
        return prompt


    def generate_response(self, prompt):
        response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=1024)["choices"][0]["text"]
        text_features = self.text_model(response)
############# NOTE: reads pre-defined input image.jpg
        #image = cv2.imread('image.jpg')
        #image = cv2.resize(image, (224, 224))
        #image_features = self.image_model(tf.keras.applications.resnet.preprocess_input(image))
        #features = text features
        return features


    def add_emotion(self, emotion):
        if np.argmax(emotion) == 0:
            return "positive"
        elif np.argmax(emotion) == 1:
            return "negative"
        else:
            return "neutral"


    def run(self):
        self.memory = ""
        self.attention = np.zeros(self.text_model.get_output_shape_at(0)[-1])
        self.emotion = np.zeros(3) # 3-dimensional vector representing positive, negative, and neutral emotions

        num_iterations = 1000
        prev_test_loss = float("inf")

        sensor_data = []
        body_actions = []
        neurons = []
        num_epochs = 10


        for i in range(num_iterations):
            prompt = self.generate_prompt("Hello, how may I assist you today?")
            response = self.generate_response(prompt)

            value = np.random.rand()
            target = np.random.rand()

            with tf.GradientTape() as tape:
                action = self.network.activate(np.concatenate((self.env._get_obs(), neurons)))
                loss = self.compute_loss(action, value, target)
                gradients = tape.gradient(loss, self.network.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))


            sensor_data.append(self.env._get_obs())
            body_actions.append(action)

            neuron_activations = []
            for neuron in neurons:
                activation = neuron.activate(sensor_data[-1])
                neuron_activations.append(activation)
            neurons = np.array(neuron_activations)

            if i % 100 == 0:
                print("Iteration {}, Loss: {}".format(i, loss.numpy()))

            if abs(loss - prev_test_loss) < 0.0001:
                break

            prev_test_loss = loss


    def compute_loss(self, action, value, target):
        loss = tf.reduce_mean(tf.square(action - target))
        return loss


class Body:
    def __init__(self):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.state = None
        self.viewer = None


    def reset(self):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.state = np.array([0, 0])


    def step(self, action):
        th, thdot = self.state       
        g = 10.
        m = 1.
        l = 1.
        dt = self.dt


        u = np.clip(action, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)  # pylint: disable=E1111
        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}


    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])


#TODO unknown class mixed with Body. Perhaps ENV or AI, it is env to augment the development of the ai and allow its growth to be controlled by the user
class ClassInsideBody:
    def __init__(self):
        self.image_model_weights = np.random.rand(2048)
        self.text_model_weights = np.random.rand(512)
    
        self.ai = AI()
        self.env = Environment()

        self.num_iterations = 1000
        self.prev_test_loss = float("inf")

        self.sensor_data = []
        self.body_actions = []
        self.neurons = []  # you need to define this properly
        self.num_epochs = 10

    def train_body1(self):
        for i in range(self.num_iterations):
            prompt = self.ai.generate_prompt("Hello, how may I assist you today?")
            response = self.ai.generate_response(prompt)

            value = np.random.rand()
            target = np.random.rand()

            with tf.GradientTape() as tape:
############# NOTE: reads pre-defined input image.jpg
                features = self.ai.text_model(prompt)
                action = ai.network.activate(np.concatenate((env._get_obs(), neurons)))
                loss = ai.compute_loss(action, value, target)
                gradients = tape.gradient(loss, ai.network.trainable_variables)
                ai.optimizer.apply_gradients(zip(gradients, ai.network.trainable_variables))


            self.sensor_data.append(self.env._get_obs())
            self.body_actions.append(action)


    def train_body1(self):
        for i in range(num_iterations):
            prompt = self.ai.generate_prompt("Hello, how may I assist you today?")
            response = self.ai.generate_response(prompt)


            value = np.random.rand()
            target = np.random.rand()

            with tf.GradientTape() as tape:
                neuron_activations = []
            for neuron in self.neurons:
                activation = neuron.activate(sensor_data[-1])
                neuron_activations.append(activation)
           
            # Append sensor data and body actions for physical body development
            self.sensor_data.append(self.env._get_obs())
            self.body_actions.append(action)


            # Print loss every 100 iterations
            if i % 100 == 0:
                print("Iteration {}, Loss: {}".format(i, loss.numpy()))
                if abs(loss - prev_test_loss) < 0.0001:
                    break

            prev_test_loss = loss


            # Compute loss and update weights
            with tf.GradientTape() as tape:
                action = self.network.activate(np.concatenate((env._get_obs(), neurons)))
                loss = self.compute_loss(action, value, target)
                gradients = tape.gradient(loss, self.network.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

            # Append sensor data and body actions for physical body development
            sensor_data.append(env._get_obs())
            body_actions.append(action)

            # Compute neuron activations
            neuron_activations = []
            for neuron in neurons:
                activation = neuron.activate(sensor_data[-1])
                neuron_activations.append(activation)
            neurons = np.array(neuron_activations)

            # Stop training if body has reached terminal state
            if env._get_obs()[0] >= 0.8:
                break
        return body_actions


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


class Environment1:
    def __init__(self):
        self.body = Body()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None


    def reset(self):
        self.state = np.random.uniform(low=-0.1, high=0.1, size=(2,))
        self.body.state = np.array([0, 0])
        self.steps_beyond_done = None
        return self._get_obs()


    def step(self, action):
        state, reward, done, info = self.body.step(action)
        self.state = state
        return state, reward, done, info


    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])


    def run_simulation(self, num_iterations):
        body_actions = self.train_body(num_iterations)


        env = self.Body()
        for i in range(num_iterations):
            action = body_actions[i]
            obs, reward, done, _ = env.step(action)
            if done:
                break


    def run(self):
        self.memory = ""
        self.attention = np.zeros(self.text_model.get_output_shape_at(0)[-1])
        self.emotion = np.zeros(3) # 3-dimensional vector representing positive, negative, and neutral emotions


        self.run_simulation(num_iterations=1000)


        print("Simulation complete.")


    def train_body(self):
        # Initialize physical body
        self.env = Body()


        # Train physical body
        inputs = np.array(sensor_data)
        targets = np.array(body_actions)
        self.env.train(inputs, targets, num_epochs=10, learning_rate=0.001)


    def run_simulation(self):
        # Initialize physical body
        self.env = Body()


        # Run simulation
        total_reward = 0
        for i in range(1000):
            action = self.network.activate(np.concatenate((self.env._get_obs(), neurons)))
            observation, reward, done, _ = self.env.step(action)
            total_reward += reward
            if done:
                break


        return total_reward


def main1():
    ai = AI()
    env = ai.env

    print("Loaded.")
    sys.stdout.flush()
    
    for i in range(10):
        observation = env.reset()
        for t in range(10):
            # Get action from AI
            print("P:")
            prompt = ai.generate_prompt("Hello, how may I assist you today?")
            response = ai.generate_response(prompt)
            action = ai.network.activate(np.concatenate((observation, response)))


            # Take action in environment
            observation, reward, done, info = env.step(action)


            # Train AI
            ai.train(response, action, reward)


            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()


def main2():
    ai = AI()
    env = ai.env


    for i in range(100):
        done = False
        env.reset()
        reward_sum = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            reward_sum += reward
            env.render()
            if done:
                print("Reward for episode {}: {}".format(i, reward_sum))
                break


    ai.run()


    def train(self, inputs, targets, num_epochs, learning_rate):
        for i in range(num_epochs):
            for j in range(len(inputs)):
                self.activate(inputs[j])
                self.backpropagate(targets[j], learning_rate)




class AI3:
    def __init__(self):
        self.text_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.image_model = hub.load("https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/5")
        self.num_neurons = 10
        self.network = NeuralNetwork(input_size=3+self.num_neurons, output_size=1, num_hidden=self.num_neurons)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


    def generate_prompt(self, prompt):
        prompt += " " + self.memory
        attention = tf.nn.softmax(tf.keras.layers.Dense(self.text_model.get_output_shape_at(0)[-1])(self.text_model(prompt) * tf.expand_dims(self.attention, axis=0)))
        prompt += " " + self.add_emotion(self.emotion)
        return prompt


    def generate_response(self, prompt):
        response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=1024)["choices"][0]["text"]
        text_features = self.text_model(response)
############# NOTE: reads pre-defined input image.jpg
        image = cv2.imread('image.jpg')
        image = cv2.resize(image, (224, 224))
        image_features = self.image_model(tf.keras.applications.resnet.preprocess_input(image))
        features = np.concatenate((text_features, image_features))
        return features


    def add_emotion(self, emotion):
        if np.argmax(emotion) == 0:
            return "positive"
        elif np.argmax(emotion) == 1:
            return "negative"
        else:
            return "neutral"


    def run(self):
        self.memory = ""
        self.attention = np.zeros(self.text_model.get_output_shape_at(0)[-1])
        self.emotion = np.zeros(3) # 3-dimensional vector representing positive, negative, and neutral emotions


        num_iterations = 1000
        prev_test_loss = float("inf")
        sensor_data = []
        body_actions = []
        neurons = []
        num_epochs = 10


        for i in range(num_iterations):
            prompt = self.generate_prompt("Hello, how may I assist you today?")
            response = self.generate_response(prompt)


            # Compute value and target
            value = np.random.rand()
            target = np.random.rand()


            # Compute loss and update weights
            with tf.GradientTape() as tape:
                action = self.network.activate(np.concatenate((self.env._get_obs(), neurons)))
                loss = self.compute_loss(action, value, target)
                gradients = tape.gradient(loss, self.network.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))


            # Append sensor data and body actions for physical body development
            sensor_data.append(self.env._get_obs())
            body_actions.append(action)


            # Compute neuron activations
            neuron_activations = []
            for neuron in neurons:
                activation = neuron.activate(sensor_data[-1])
                neuron_activations.append(activation)
            neurons = np.array(neuron_activations)


            # Print loss every 100 iterations
            if i % 100 == 0:
                print("Iteration {}, Loss: {}".format(i, loss.numpy()))


            # Stop training if loss doesn't improve significantly
            if abs(loss - prev_test_loss) < 0.0001:
                break


            prev_test_loss = loss


        # Train physical body


    def run_simulation(self):
        self.env = self.Body()


        # Run simulation for 100 steps
        for i in range(100):
            # Generate prompt and response
            prompt = self.generate_prompt("Hello, how may I assist you today?")
            response = self.generate_response(prompt)


            # Compute action using neural network and body state
            action = self.network.activate(np.concatenate((self.env._get_obs(), self.neurons)))


            # Step body in the environment
            obs, reward, done, info = self.env.step(action)


            # Append sensor data and body actions for physical body development
            self.sensor_data.append(obs)
            self.body_actions.append(action)


            # Compute neuron activations
            neuron_activations = []
            for neuron in self.neurons:
                activation = neuron.activate(self.sensor_data[-1])
                neuron_activations.append(activation)
            self.neurons = np.array(neuron_activations)


            # Print current step
            print("Step:", i)


    def visualize(self):
        # Create video writer object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('simulation.mp4', fourcc, 60.0, (640, 480))


        # Set up body viewer
        self.env.viewer = self.env._get_viewer()
        self.env.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
        self.env.viewer.set_camera(camera_id=0)


        # Run simulation and save frames to video file
        for i in range(len(self.sensor_data)):
            obs = self.sensor_data[i]
            action = self.body_actions[i]
            self.env.state = np.array([np.arctan2(obs[1], obs[0]), obs[2]])
            img = self.env.render()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.putText(img, "Action: {:.2f}".format(action[0]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            video_writer.write(img)


        # Release video writer and body viewer
        video_writer.release()
        self.env.viewer.finish()


class Environment:
    def __init__(self):
        self.body = Body()
        self.state = None
        self.max_steps = 1000
        self.num_steps = 0


    def reset(self):
        self.num_steps = 0
        high = np.array([np.pi, 1])
        self.state = np.random.uniform(low=-high, high=high)
        return self._get_obs()


    def step(self, action):
        self.num_steps += 1
        self.state, reward, done, info = self.body.step(action)
        if self.num_steps >= self.max_steps:
            done = True
        return self._get_obs(), reward, done, info


    def _get_obs(self):
        return self.state


def main3():
    ai = AI()
    env = Environment()


    num_iterations = 1000
    prev_test_loss = float("inf")


    for i in range(num_iterations):
        prompt = ai.generate_prompt("Hello, how may I assist you today?")
        response = ai.generate_response(prompt)


        # Compute value and target
        value = np.random.rand()
        target = np.random.rand()


        # Compute loss and update weights
        with tf.GradientTape() as tape:
            action = ai.network.activate(np.concatenate((env._get_obs(), ai.neurons)))
            loss = ai.compute_loss(action, value, target)
            gradients = tape.gradient(loss, ai.network.trainable_variables)
            ai.optimizer.apply_gradients(zip(gradients, ai.network.trainable_variables))


        # Append sensor data and body actions for physical body development
        ai.sensor_data.append(env._get_obs())
        ai.body_actions.append(action)


        # Compute neuron activations
        neuron_activations = []
        for neuron in ai.neurons:
            activation = neuron.activate(ai.sensor_data[-1])
            neuron_activations.append(activation)
        ai.neurons = np.array(neuron_activations)


        # Print loss every 100 iterations
        if i % 100 == 0:
            print("Iteration {}, Loss: {}".format(i, loss.numpy()))


        # Stop training if loss doesn't improve significantly
        if abs(loss - prev_test_loss) < 0.0001:
            break


        prev_test_loss = loss


if __name__ == '__main__':
    main1()
