import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from tqdm import tqdm
import subprocess
from matplotlib.colors import Normalize

np.random.seed(10)
data21 = loadmat('data21.mat')
data22 = loadmat('data22.mat')

#!Loading Weights And Biases From .mat File
A_1 = np.array(data21['A_1'])
A_2 = np.array(data21['A_2'])
B_1 = np.array(data21['B_1'])
B_2 = np.array(data21['B_2'])


#! Loading X_i, X_n
X_i = np.array(data22['X_i'])
X_n = np.array(data22['X_n'])


def u(Z):
    #! Generator producing handwritten eights
    W1 = A_1 @ Z + B_1
    Z1 = np.maximum(0, W1)

    W2 = A_2 @ Z1  + B_2
    X = 1 / (1 + np.exp(W2))

    return X

def plot(X):
    fig, axes = plt.subplots(2, 4, figsize=(8, 8))
    print(X_i.shape)
    X = np.concatenate((X_i, X), axis=1)
    for i, ax in enumerate(axes.flat):
        ax.imshow((X[:,i]).reshape(28,28).T, cmap='gray')
        ax.axis('off')  # Turn off axis labels
    plt.show()



def gradient(T,Z,N, distored_data, M):

    W1 = A_1 @ Z + B_1
    Z1 = np.maximum(0, W1)

    W2 = A_2 @ Z1  + B_2
    X = 1 / (1 + np.exp(W2))


    res = distored_data.reshape(N,1) - np.dot(T,X)
    norm_squared = np.linalg.norm(res) ** 2

    
    u2 = 2 * np.dot(-T.T, res) / norm_squared

    v2 = u2 * (X * (X - 1)) #? Sigmoid Rerivative Based On Output X

    u1 = A_2.T @ v2


    v1 = u1 * np.where(W1 > 0, 1, 0)

    u0 = A_1.T @ v1

    nablaJ_z = M * u0 + 2 * Z
    return nablaJ_z, norm_squared


def minimize(Z, n, N, iterations):
    
    I = np.eye(N)
    zeros = np.zeros((N, 784 - N))
    
    T = np.concatenate((I, zeros), axis=1) #! Matrix T
    
    distorted_data = X_n[:,n][:N]

    M = len(distorted_data)
    
    m = 0.01
    l = 0.01


    c = 10**(-8)
    nablaZ, norm_squared = gradient(T, Z,N, distorted_data, M) 

    J = 0
    px =  (nablaZ) ** 2

    cost = []

    pbar = tqdm(range(iterations), colour='blue', position=0, desc=f"Restoring Image... Cost = {J}")

    for _ in pbar:
    #? -------------- Initiating Gradient Descent -------------------

        J = M * np.log(norm_squared) + np.linalg.norm(Z)**2 #? Calculating Cost

        Z -= m * nablaZ / np.sqrt(px + c)

        pbar.set_description(f"Restoring Image... Cost = {J}")
        nablaZ, norm_squared = gradient(T, Z,N, distorted_data, M)        

        px = (1 - l) * px + l * (nablaZ) ** 2
        cost.append(J)


    X = u(Z)
    padded = np.pad(distorted_data, (0, 784 - len(distorted_data)), 'constant').reshape(784,1)
    return X, padded, cost


def plot_images(original_images, distorted_images, recovered_images,N):
    fig, axes = plt.subplots(len(distorted_images), 3, figsize=(8, 8))

    for i in range(len(distorted_images)):
   
        
        axes[i, 0].imshow(original_images[:,i].reshape(28,28).T, cmap='gray')
        axes[i, 0].axis('off')  
               
        norm = Normalize(vmin=0, vmax=np.max(distorted_images[i]))
        axes[i, 1].imshow(distorted_images[i].reshape(28,28).T, cmap='gray', norm =norm)
        axes[i, 1].axis('off')  
        
        
        axes[i, 2].imshow(recovered_images[i].reshape(28,28).T, cmap='gray')
        axes[i, 2].axis('off')  

    plt.tight_layout()  
    plt.savefig('recovered'+str(N)+'.png')

    plt.show()

def plot_costs(costs,N):
    fig, axes = plt.subplots(len(costs), 1, figsize=(8, 8))

    for i in range(len(costs)):
        axes[i].plot(np.linspace(1,len(costs[i]), len(costs[i])), costs[i])
    
    plt.tight_layout()  # Adjust the layout to avoid overlap
    plt.savefig('costs'+str(N)+'.png')
    plt.show()
   
def main():
    Z = np.random.normal(0,1, size=(10,1))

    iterations = 3000
    N = 500
    subprocess.run('clear')
    
    #! Running recovering algorithm for each image of X_n (keeping the first )
    X1, distorted_data1, cost1 = minimize(Z,0, N, iterations) 
    X2, distorted_data2, cost2 = minimize(Z,1, N, iterations)
    X3, distorted_data3, cost3 = minimize(Z,2, N, iterations)
    X4, distorted_data4, cost4 = minimize(Z,3, N, iterations)

    distorted_images = [distorted_data1, distorted_data2, distorted_data3, distorted_data4]
    recovered_images = [X1, X2, X3, X4]
    plot_images(X_i, distorted_images, recovered_images, N)
    
    costs = [cost1, cost2, cost3, cost4]
    plot_costs(costs, N)

if __name__ == "__main__":
    main()

        