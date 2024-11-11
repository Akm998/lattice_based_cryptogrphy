import numpy as np
import matplotlib.pyplot as plt

# Function to generate keys for the LWE cryptosystem
def generate_keys(n, q, std_dev, secret_key_length=1):
    """ Generate keys for the LWE cryptosystem.
    
    Args:
        n (int): The dimension of the secret key vector.
        q (int): The modulus.
        std_dev (float): Standard deviation for the error distribution.
        secret_key_length (int): Length of the secret key.
    
    Returns:
        tuple: Returns (secret_key, public_key), where `secret_key` is a vector
               and `public_key` is a matrix whose columns correspond to
               (a_i, b_i) where b_i = <a_i, s> + e_i (mod q).
    """
    # Secret key is a vector in Z_q^n
    secret_key = np.random.randint(low=0, high=q, size=(n, secret_key_length))
    
    # Public key generation
    A = np.random.randint(low=0, high=q, size=(n, n))
    e = np.random.normal(loc=0.0, scale=std_dev, size=(n, secret_key_length)).astype(int)
    B = (A @ secret_key + e) % q
    
    public_key = (A, B)
    return secret_key, public_key

# Function to encrypt a message using the public key
def encrypt(public_key, message, q):
    """ Encrypt a message using the public key.
    
    Args:
        public_key (tuple): The public key (A, B) matrix pair.
        message (int): The message to encrypt, must be 0 or 1.
        q (int): The modulus.
    
    Returns:
        np.array: The encrypted message.
    """
    A, B = public_key
    n = A.shape[0]
    u = np.random.randint(0, 2, size=(n, 1))  # Random binary vector
    c1 = (A.T @ u) % q
    c2 = (B.T @ u + message * (q // 2)) % q
    
    return c1, c2  # Return c1 and c2 separately

# Function to decrypt a message using the secret key
def decrypt(secret_key, ciphertext, q):
    """ Decrypt a message using the secret key.
    
    Args:
        secret_key (np.array): The secret key.
        ciphertext (tuple): The ciphertext tuple (c1, c2).
        q (int): The modulus.
    
    Returns:
        int: The decrypted message.
    """
    c1, c2 = ciphertext
    message_rec = (c2 - secret_key.T @ c1) % q
    message_rec = message_rec[0][0]  # Assuming single dimension secret key
    return np.round(message_rec / (q // 2)) % 2

# Function to load an image and preprocess it
def load_and_preprocess_image(image_path, threshold=128, resize_factor=10):
    img = plt.imread(image_path)
    # Convert the image to grayscale
    img_gray = np.mean(img, axis=2)
    # Resize the image
    img_resized = img_gray[::resize_factor, ::resize_factor]
    # Binarize the image based on a threshold
    img_bin = (img_resized > threshold).astype(int)
    return img_bin

# Load and preprocess the image
image_path = r"C:/Users/ankan/OneDrive/Pictures/Saved Pictures/Batman.jpg"
image = load_and_preprocess_image(image_path)

# Example usage
n = image.size  # Dimension of the secret (based on the resized image size)
q = 101  # Modulus, a prime number
std_dev = 7  # Standard deviation of the error term

# Generate keys
s, pk = generate_keys(n, q, std_dev)

# Encrypt the image
ciphertext = encrypt(pk, image.ravel(), q)

# Decrypt the image
decrypted_image = decrypt(s, ciphertext, q).reshape(image.shape)

print("image:", decrypted_image)

# Plot the original and decrypted images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(decrypted_image, cmap='gray')
plt.title('Decrypted Image')

plt.show()
