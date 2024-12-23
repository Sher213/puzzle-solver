import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import base64
import random
import copy
import pandas as pd

#Debug Constant
DB = False

#Background Color (Grey)
BG_COL = (90, 90, 90)

#Smallest pic resolution (width or height) to save
MIN_PIC_TO_SAVE = 200

def generate_puzzle_matrix(rows, cols):
    matrix = np.zeros((rows, cols, 4), dtype=int)

    for i in range(rows):
        for j in range(cols):
            left = 0 if j == 0 else -matrix[i, j-1, 2]  
            top = 0 if i == 0 else -matrix[i-1, j, 3]  
            right = random.choice([-1, 1]) if j < cols - 1 else 0  
            bottom = random.choice([-1, 1]) if i < rows - 1 else 0  

            matrix[i, j] = [left, top, right, bottom]

    return matrix

def split_image(image_path, rows, cols):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    piece_height = h // rows
    piece_width = w // cols

    pieces = []
    for i in range(rows):
        row_pieces = []
        for j in range(cols):
            piece = image[i * piece_height:(i + 1) * piece_height, j * piece_width:(j + 1) * piece_width]
            row_pieces.append(piece)
        pieces.append(row_pieces)

    return pieces

def modify_piece_with_tabs(piece, measurements, direction, neighbor_piece=None, action=0, Debug=False):
    """
    Modify the piece to add or remove tabs derived from the neighbor piece.
    - direction: The side to modify ('left', 'top', 'right', 'bottom').
    - neighbor_piece: The adjacent piece for reference (mandatory for tabs).
    - action: 1 to add a tab, -1 to remove a tab, 0 to leave as is.
    """
    if neighbor_piece is None or action == 0:
        return piece  # No modification needed

    tab_height = measurements[0]
    tab_width = measurements[1]

    tab_height_factor = int(tab_height // 1.5)
    tab_width_factor = int(tab_width // 0.75)

    h = measurements[2]
    w = measurements[3]

    left = tab_width
    top = tab_height
    right = tab_width + w
    bottom = tab_height + h

    if direction == 'right' and action == 1:
        # Add a tab from the left of the neighbor piece
        piece[top+(tab_height_factor):bottom-(tab_height_factor), right:] = neighbor_piece[(tab_height_factor):h-(tab_height_factor), :tab_width]
    elif direction == 'bottom' and action == 1:
        # Add a tab from the top of the neighbor piece
        piece[bottom:bottom+tab_height, left+(tab_width_factor):right-(tab_width_factor)] = neighbor_piece[:tab_height, (tab_width_factor):w-(tab_width_factor)]
        
        if Debug == True:
            print("Neighbor Piece: Bottom")
            plt.imshow(neighbor_piece)
            plt.axis('off')
            plt.show()
    elif direction == 'left' and action == 1:
        # Add a tab from the right of the neighbor piece
        piece[top+(tab_height_factor):bottom-(tab_height_factor), :left] = neighbor_piece[(tab_height_factor):h-(tab_height_factor), w-tab_width:w]
        #print(top+(tab_height_factor),bottom-(tab_height_factor), left,(tab_height_factor),h-(tab_height_factor), w-tab_width,w)
    elif direction == 'top' and action == 1:
        # Add a tab from the bottom of the neighbor piece
        piece[:top, left+(tab_width_factor):right-(tab_width_factor)] = neighbor_piece[:tab_height, (tab_width_factor):w-(tab_width_factor)]
    
    elif direction == 'right' and action == -1:
        #piece = piece[:, :right]
        piece[top+(tab_height_factor):bottom-(tab_height_factor), right-tab_width:right] = BG_COL

        if Debug == True:
            print("Right")
            plt.imshow(piece)
            plt.axis('off')
            plt.show()
    elif direction == 'bottom' and action == -1:
        #piece = piece[:bottom, :]
        piece[bottom-tab_height:bottom, left+(tab_width_factor):right-(tab_width_factor)] = BG_COL
        
        if Debug == True:
            print("Bottom")
            plt.imshow(piece)
            plt.axis('off')
            plt.show()
    elif direction == 'left' and action == -1:
        #piece = piece[:, left:]
        piece[top+(tab_height_factor):bottom-(tab_height_factor), left:left+tab_width] = BG_COL
        
        if Debug == True:
            print("Left")
            plt.imshow(piece)
            plt.axis('off')
            plt.show()
    elif direction == 'top' and action == -1:
        #piece = piece[top:, :]
        piece[top:top+tab_height, left+(tab_width_factor):right-(tab_width_factor)] = BG_COL
        
        if Debug == True:
            print("Top")
            plt.imshow(piece)
            plt.axis('off')
            plt.show()

    return piece

def apply_matrix_to_pieces(pieces, puzzle_matrix):
    rows, cols, _ = puzzle_matrix.shape
    pieces_copy = copy.deepcopy(pieces)

    for i in range(rows):
        for j in range(cols):
            piece = pieces[i][j]
            left, top, right, bottom = puzzle_matrix[i, j]

            h, w, _ = piece.shape
            tab_width = w // 5
            tab_height = h // 5
            measurments = [tab_height, tab_width, h, w]

            resize_piece = np.full((h + 2 * tab_height, w + 2 * tab_width, 3), BG_COL, dtype=np.uint8)
            resize_piece[tab_height:tab_height+h, tab_width:tab_width+w ] = piece

            if right != 0 and j < cols - 1:  
                pieces[i][j] = modify_piece_with_tabs(resize_piece, measurments, 'right', pieces_copy[i][j + 1], action=right, Debug=DB)
            if bottom != 0 and i < rows - 1:  
                pieces[i][j] = modify_piece_with_tabs(resize_piece, measurments, 'bottom', pieces_copy[i + 1][j], action=bottom, Debug=DB)
            if left != 0 and j > 0:  
                pieces[i][j] = modify_piece_with_tabs(resize_piece, measurments, 'left', pieces_copy[i][j - 1], action=left, Debug=DB)
            if top != 0 and i > 0:  
                pieces[i][j] = modify_piece_with_tabs(resize_piece, measurments, 'top', pieces_copy[i - 1][j], action=top, Debug=DB)

    return pieces

def display_puzzle(pieces):
    """Display the puzzle pieces using matplotlib."""
    rows = len(pieces)
    cols = len(pieces[0])

    # Assemble the image back from the puzzle pieces
    assembled_image = np.vstack([np.hstack(row) for row in pieces])

    # Shuffle the pieces
    flat_pieces = [piece for row in pieces for piece in row]  # Flatten the list of pieces
    random.shuffle(flat_pieces)  # Shuffle the pieces
    
    # Rebuild the shuffled pieces into a 2D grid (same dimensions as the original puzzle)
    shuffled_pieces = []
    for i in range(rows):
        shuffled_pieces.append(flat_pieces[i * cols:(i + 1) * cols])  # Rebuild rows

    # Assemble the image back from the shuffled puzzle pieces
    shuffled_image = np.vstack([np.hstack(row) for row in shuffled_pieces])

    # Show the assembled image
    plt.imshow(assembled_image)
    plt.show()
    plt.imshow(shuffled_image)
    plt.axis('off')
    plt.show()

def save_puzzle_images_cv2(pieces, assembled_path="assembled_image.png", shuffled_path="shuffled_image.png"):
    """
    Save the puzzle pieces as images in their original and shuffled states using OpenCV.

    Args:
        pieces (list of list of np.ndarray): 2D list of image pieces.
        assembled_path (str): File path to save the assembled image.
        shuffled_path (str): File path to save the shuffled image.
    """
    rows = len(pieces)
    cols = len(pieces[0])

    # Assemble the image back from the puzzle pieces
    assembled_image = np.vstack([np.hstack(row) for row in pieces])

    # Flatten and shuffle the pieces
    flat_pieces = [piece for row in pieces for piece in row]
    random.shuffle(flat_pieces)

    # Rebuild the shuffled pieces into a 2D grid
    shuffled_pieces = []
    for i in range(rows):
        shuffled_pieces.append(flat_pieces[i * cols:(i + 1) * cols])

    # Assemble the shuffled image
    shuffled_image = np.vstack([np.hstack(row) for row in shuffled_pieces])

    # Convert images to uint8 if needed (OpenCV requires uint8 images)
    if assembled_image.dtype != np.uint8:
        # Ensure the image is in the correct range [0, 255] before converting
        assembled_image = np.clip(255 * assembled_image, 0, 255).astype(np.uint8)
    if shuffled_image.dtype != np.uint8:
        shuffled_image = np.clip(255 * shuffled_image, 0, 255).astype(np.uint8)

    # Check dimensions and save only if either height or width exceeds min pixels
    if assembled_image.shape[0] > MIN_PIC_TO_SAVE or assembled_image.shape[1] > MIN_PIC_TO_SAVE:  # Height or Width of assembled_image
        # Convert from RGB to BGR if needed (OpenCV uses BGR format)
        if len(assembled_image.shape) == 3 and assembled_image.shape[2] == 3:  # Check if it's a color image
            assembled_image = cv2.cvtColor(assembled_image, cv2.COLOR_RGB2BGR)
            shuffled_image = cv2.cvtColor(shuffled_image, cv2.COLOR_RGB2BGR)

        # Save images
        cv2.imwrite(assembled_path, assembled_image)
        cv2.imwrite(shuffled_path, shuffled_image)
        return True
    
    return False

def image_to_base64(image):
    """Convert an image to base64 encoding."""
    _, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64

def file_to_base64(image_path):
    """Convert an image file to base64 encoding."""
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    return image_base64

def create_puzzles_and_df(df, image_path, folder_substr, rows, cols):
    # Step 1: Generate puzzle matrix
    puzzle_matrix = generate_puzzle_matrix(rows, cols)
    #print("Puzzle Matrix:")
    #print(puzzle_matrix)

    # Step 2: Split the image into pieces
    pieces = split_image(image_path, rows, cols)

    # Step 3: Modify pieces based on the puzzle matrix
    pieces = apply_matrix_to_pieces(pieces, puzzle_matrix)

    # Step 4: Display the puzzle pieces
    #display_puzzle(pieces)

    #Step 5: Save the assembled and the shuffled pictures
    assembled_path = os.path.join("./assembled", f"{folder_substr}-assembled.png")
    shuffled_path = os.path.join("./shuffled", f"{folder_substr}-shuffled.png")
    saved = save_puzzle_images_cv2(pieces, assembled_path, shuffled_path)
    
    if saved:
        # Convert images to base64
        assembled_base64 = file_to_base64(assembled_path)
        shuffled_base64 = file_to_base64(shuffled_path)
        original_base64 = file_to_base64(image_path)

        # Step 6: Populate the DataFrame
        single_piece_width, single_piece_height = pieces[0][0].shape[1], pieces[0][0].shape[0]
        new_entry = pd.DataFrame([{
            "name": folder_substr,
            "original_image": original_base64,  # Store the base64-encoded original image
            "assembled_puzzle": assembled_base64,
            "shuffled_puzzle": shuffled_base64,
            "puzzle_row_col": rows,
            "single_piece_width": single_piece_width,
            "single_piece_height": single_piece_height
        }])

        df = pd.concat([df, new_entry], ignore_index=True)
    
    return df


#Perform function
df = pd.DataFrame(columns=["name", "original_image", "assembled_puzzle", "shuffled_puzzle", "puzzle_row_col", "single_piece_width", "single_piece_height"])

root_path = "../puzzles/Download-From-Google-Images/"
for dir in os.listdir(root_path):
    if not ("git" in dir or "download" in dir or "README" in dir):
        for filename in os.listdir(os.path.join(root_path, dir)):
            if not ".gif" in filename: 
                image_path = os.path.join(root_path, dir, filename)
                print(image_path)
                row_col = random.randrange(4, 9)
                df = create_puzzles_and_df(df, image_path, image_path.split(chr(92))[-1].split('.')[0], row_col, row_col)
                print(df)

df.to_csv("images-and-puzzles.csv")