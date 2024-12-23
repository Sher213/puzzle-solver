import random
import os
import shutil
import time
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#Set constants
MAX_FEATURES = 10500
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
PIECE_CENTRE_PAD = 25

PIECES_IMAGE_PATH = './pieces'

class RegressionDataGenerator(Sequence):
    def __init__(self, dataframe, batch_size, img_size, shuffle=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataframe))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_df = self.dataframe.iloc[batch_indexes]

        images = []
        targets = []

        for _, row in batch_df.iterrows():
            img_path = row['puzzle_img_path']
            target_value = row['position_normalized']

            # Load and process the image
            img = cv2.imread(f"{img_path}")
            if img is None:
                continue  # Skip failed images
            img = cv2.resize(img, self.img_size)  # Resize image
            img = img / 255.0  # Normalize image

            images.append(img)
            targets.append(target_value)

        return np.array(images), np.array(targets)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# --------------- Step 1: Process Images Functions -----------------
def split_data(input_dir, output_dir="./data/images/", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Splits image data into training, validation, and testing subsets.

    Args:
        input_dir (str): Path to the directory containing class folders with images.
        output_dir (str): Path to save the split dataset (training, validation, testing).
        train_ratio (float): Proportion of data for training (default 70%).
        val_ratio (float): Proportion of data for validation (default 20%).
        test_ratio (float): Proportion of data for testing (default 10%).

    Raises:
        ValueError: If the ratios do not sum to 1.0.
    """
    # Ensure the split ratios sum to 1.0
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0.")

    image_files = [os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, fname))]

    # Create output directories if they don't exist
    for subset in ['train', 'val', 'test']:
        subset_dir = os.path.join(output_dir, subset)
        os.makedirs(subset_dir, exist_ok=True)

        # Split into train, validation, and test sets
        train_files, temp_files = train_test_split(image_files, test_size=(val_ratio + test_ratio), random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

        # Copy images to their respective directories
        for subset, files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
            subset_class_dir = os.path.join(output_dir, subset)
            os.makedirs(subset_class_dir, exist_ok=True)
            for file_path in files:
                try:
                    shutil.copy(file_path, subset_class_dir)
                    print(f"Copied {file_path} to {subset_class_dir}")
                except:
                    continue
            
    print(f"Data split into training, validation, and testing subsets in {output_dir}.")

def gen_train_test_gen():
    # Step 1: Load CSV file with image paths and regression targets
    csv_file = './puzzle-with-positions.csv'  # Path to your CSV
    df = pd.read_csv(csv_file)

    # Step 2: Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Step 4: Create the training and testing data generators
    train_generator = RegressionDataGenerator(
        train_df, batch_size=BATCH_SIZE, img_size=IMG_SIZE,
    )

    test_generator = RegressionDataGenerator(
        test_df, batch_size=BATCH_SIZE, img_size=IMG_SIZE,
    )

    return train_generator, test_generator

# ------------------ Extract Tiles from Image ------------------
def extract_tiles(image_path, grid_size=(6, 8)):
    """Divide an image into grid_size rows and columns."""
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    grid_h, grid_w = grid_size

    tile_height = h // grid_h
    tile_width = w // grid_w

    tiles = []
    for row in range(grid_h):
        for col in range(grid_w):
            x1, y1 = col * tile_width, row * tile_height
            x2, y2 = x1 + tile_width, y1 + tile_height
            tile = image[y1:y2, x1:x2]
            tiles.append(tile)
    return tiles, (tile_height, tile_width)

# ------------------ Centre Image ------------------
def center_image_on_background(image, background_size, background_color=(90, 90, 90)):
    """
    Centers the given image on a background of specified size and color.
    
    Args:
    - image: The image to center (NumPy array).
    - background_size: Tuple of (height, width) for the background.
    - background_color: The color of the background (default is white).
    
    Returns:
    - result: The background with the centered image.
    """
    # Get the dimensions of the background
    background_height, background_width = background_size
    
    # Get the dimensions of the image
    img_height, img_width = image.shape[:2]
    
    # Create the background (solid color)
    background = np.full((background_height, background_width, 3), background_color, dtype=np.uint8)
    
    # Calculate the position to place the image in the center
    top_left_y = (background_height - img_height) // 2
    top_left_x = (background_width - img_width) // 2

    # Place the image on the background
    background[top_left_y:top_left_y+img_height, top_left_x:top_left_x+img_width] = image

    return background

# ------------------ Reconstruct Image ------------------
def reconstruct_image(predictions, shuffled_tiles, grid_size, tile_size):
    """Reconstruct the image using the predicted positions."""
    grid_h, grid_w = grid_size
    tile_h, tile_w = tile_size

    # Create a blank canvas to reconstruct the image
    reconstructed = np.zeros((grid_h * tile_h, grid_w * tile_w, 3), dtype=np.uint8)

    # Place each tile in the predicted position
    for idx, pos in enumerate(predictions):
        row, col = divmod(pos, grid_w)
        y1, x1 = row * tile_h, col * tile_w
        y2, x2 = y1 + tile_h, x1 + tile_w
        reconstructed[y1:y2, x1:x2] = shuffled_tiles[idx]
    return reconstructed

# ------------------ Extract Features ------------------
def extract_features(tiles):
    """Extract simple features by flattening the tiles."""
    features = [tile.flatten() for tile in tiles]
    features = [item for sublist in features for item in sublist]
    return features

# ------------------ Step X: Train a Classifier ------------------
def train_classifier(ordered_features, labels):
    """Train a RandomForestClassifier with the ordered puzzle tiles."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(ordered_features, labels)
    return clf

# ------------------ Step 2: Make and Train CNN ------------------
def make_and_train_cnn(train_gen, val_gen):
    # Make CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(350, 350, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Use Global Average Pooling instead of Flatten
        GlobalAveragePooling2D(),

        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1)  # Output layer without activation (regression)
    ])

    # Compile Model
    model.compile(
        optimizer=Adam(learning_rate=0.0001), # Adam optimizer for better convergence
        loss='mean_squared_error',          # MSE is commonly used for regression
        metrics=['mae']                     # Mean Absolute Error is useful for regression evaluation
    )

    # Train Model
    history = model.fit(
        train_gen,                          # Training generator
        validation_data=val_gen,            # Validation generator
        epochs=25,                           # Number of epochs
        workers=4,                           # Parallel data loading workers (adjust as needed)
        use_multiprocessing=True            # Use multiprocessing for better parallel data loading
    )

    return model, history

# ------------------ Step 3: Evaluate Model Functions ------------------
def predict_positions(clf, shuffled_features):
    """Predict the positions of the shuffled tiles."""
    predictions = clf.predict(shuffled_features)
    return predictions

def evaluate_model(model):
    #Get test data
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_data = test_datagen.flow_from_directory(
        "./data/images/test/",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    # Evaluate
    test_loss, test_accuracy = model.evaluate(test_data)
    print(f"Test Accuracy: {test_accuracy}")

# ------------------ Create CSVs ------------------
def create_csv_from_shuffled_puzzle_path():
    #paths for files
    puzzle_path = "./puzzle-maker/shuffled/"
    csv_path = "./puzzle-numberer/puzzles-numbered-csv/"

    #get directories
    puzzles_dir = sorted(os.listdir(puzzle_path))
    csv_dir = sorted(os.listdir(csv_path))
    
    '''#DEBUG
    s = []'''

    #initialize the outgoing dataframe with columns
    columns = [f"flattened_from_image {i}" for i in range(MAX_FEATURES)] + ["position_in_puzzle"]
    df_out = pd.DataFrame(columns=columns)
    total_iterations = 0
    #iterate through files together
    for file_index, (puzzle, csv) in enumerate(zip(puzzles_dir, csv_dir)):
        total_iterations+=1
        #Full path, including file name
        full_puzzle_path = os.path.join(puzzle_path, puzzle)
        full_csv_path = os.path.join(csv_path, csv)

        print(full_puzzle_path)

        #Read csv and initialize the indexer and outgoing dataframe
        df_in = pd.read_csv(full_csv_path)
        df_index = 0

        #set row_col (rows and columns)
        row_col = df_in["row_col"][0]

        #extract tiles
        shuffled_tiles, tile_size = extract_tiles(full_puzzle_path, (row_col, row_col))
        
        #iterate through tiles
        for i, tile in enumerate(shuffled_tiles):
            total_iterations+=1
            print(f"df_out num elements: {df_out.size}")
            if df_out.size >= 500000:
                df_out.to_csv(f"./tiled-numbered{total_iterations}.csv")
                df_out = pd.DataFrame(columns=columns)

            if len(shuffled_tiles) == row_col**2 == len(df_in["row_col"]):
                #get features of each tile by flattening it so that it can be added to dataframe, get position of tile in puzzle
                features = extract_features(tile)
                position = df_in.iloc[df_index]['position_in_puzzle']

                '''#DEBUG
                print(len(features))
                if df_out.empty:
                    s.append(len(features))'''

                #get number of features extracted
                num_features_extraced = len(features)

                #if the number of features is less than MAX_FEATURES (or MAX_FEATURES), fill the missing values with the first flattened value
                if num_features_extraced <= MAX_FEATURES:
                    #gather data, make to a dataframe and add it to outgoing dataframe
                    data = list(features + [90 for i in range(MAX_FEATURES - num_features_extraced)]) + [position]
                    new_row = pd.DataFrame([data], columns=columns)
                    df_out = pd.concat([df_out, new_row], ignore_index=True)
                #else if > MAX_FEATURES, keep only the first MAX_FEATURES features
                elif num_features_extraced > MAX_FEATURES:
                    data = list(features[:MAX_FEATURES]) + [position]
                    new_row = pd.DataFrame([data], columns=columns)
                    df_out = pd.concat([df_out, new_row], ignore_index=True)
        
            df_index+=1

    '''#DEBUG
    print(f"Feature Max: {max(s)}, Feature Min: {min(s)}, Feature Avg: {sum(s) / len(s)}" )'''

def save_pieces_and_to_csv():
    #paths for files
    puzzle_path = "./puzzle-maker/shuffled/"
    csv_path = "./puzzle-numberer/puzzles-numbered-csv/"

    #make the pieces dir
    os.makedirs(PIECES_IMAGE_PATH, exist_ok=True)

    #get directories
    puzzles_dir = sorted(os.listdir(puzzle_path))
    csv_dir = sorted(os.listdir(csv_path))

    #initialize the outgoing dataframe with columns
    columns = ["puzzle_img_path", "position_normalized"]
    df_out = pd.DataFrame(columns=columns)
    
    #iterate through files together
    for file_index, (puzzle, csv) in enumerate(zip(puzzles_dir, csv_dir)):
        #Full path, including file name
        full_puzzle_path = os.path.join(puzzle_path, puzzle)
        full_csv_path = os.path.join(csv_path, csv)

        print(full_puzzle_path)

        #Read csv and initialize the indexer and outgoing dataframe
        df_in = pd.read_csv(full_csv_path)
        df_index = 0

        #set row_col (rows and columns)
        row_col = df_in["row_col"][0]

        #extract tiles
        shuffled_tiles, tile_size = extract_tiles(full_puzzle_path, (row_col, row_col))
        
        #iterate through tiles
        for i, tile in enumerate(shuffled_tiles):
            if len(shuffled_tiles) == row_col**2 == len(df_in["row_col"]):

                position_norm = df_in.iloc[df_index]['position_normalized']

                r = random.randint(1, 1000000000)
                output_path = f"{PIECES_IMAGE_PATH}/puzzle{r}.png"

                print(f"Centering piece 'puzzle{r}.png.'")
                centered_tile = center_image_on_background(tile, (tile_size[0] + PIECE_CENTRE_PAD, tile_size[1] + PIECE_CENTRE_PAD))

                if cv2.imwrite(output_path, centered_tile):
                    df_out = pd.concat([df_out, pd.DataFrame([{"puzzle_img_path":  output_path, "position_normalized" : position_norm}])])
                    print(f"Saved {output_path}.")

                else:
                    print("Failed to save the image.")
            
            df_index+=1
        
    df_out.to_csv(f"./puzzle-with-positions.csv")
    
def main():
    #Manage GPU memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    #This function
    # 1. Splits the puzzles up into pieces
    # 2. Centres the pieces
    # 3. Saves each piece as an individual image
    # 4. Saves the pieces path to a csv along with its normalized position
    print("Saving pieces...")
    #save_pieces_and_to_csv()

    #split the test data into test and train dirs then get the data as train and evaluation data
    print("Splitting the pieces into train, val and test.")
    #split_data(input_dir=f"{PIECES_IMAGE_PATH}/")
    print("Splitting done.")
    print("...")
    train_gen, val_gen = gen_train_test_gen()

    print("Generated train and val data.")

    print("Training model...")
    #model, history = make_and_train_cnn(train_gen, val_gen)
    print("Model Trained.")

    print("Saving model.")
    #model.save("cnn_model_puzzle_predictor.h5")
    print("Model saved.")

    print("Loading model.")
    # Load the model
    model = load_model("cnn_model_puzzle_predictor.h5")

    evaluate_model(model)

if __name__ == "__main__":
    main()
