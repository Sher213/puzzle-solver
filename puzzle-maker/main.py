import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

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

def modify_piece_with_tabs(piece, measurements, direction, neighbor_piece=None, action=0):
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

    h = measurements[2]
    w = measurements[3]

    left = tab_width
    top = tab_height
    right = tab_width + w
    bottom = tab_height + h

    if direction == 'right' and action == 1:
        # Add a tab from the left of the neighbor piece
        #resize_piece[piece[top+(tab_height // 2):h-(tab_height//2), right:] = resize_neighbor_piece[top+(tab_height // 2):h-(tab_height//2), :tab_width]
        #print (piece[top+(tab_height // 2):h-(tab_height//2), right:], top+(tab_height // 2), tab_height, right)
        print(top+(tab_height//2), bottom-(tab_height//2), (tab_height // 2), h-(tab_height//2))
        piece[top+(tab_height // 2):bottom-(tab_height//2), right:] = neighbor_piece[(tab_height // 2):h-(tab_height//2), -tab_width:]
    elif direction == 'bottom' and action == 1:
        # Add a tab from the top of the neighbor piece
        piece[bottom:, left+(tab_width // 2):right-(tab_width //2)] = neighbor_piece[:tab_height, (tab_width // 2):w-(tab_width//2)]
    elif direction == 'left' and action == 1:
        # Add a tab from the right of the neighbor piece
        piece[top+(tab_height // 2):bottom-(tab_height//2), :left] = neighbor_piece[(tab_height // 2):h-(tab_height//2), :tab_width]
    elif direction == 'top' and action == 1:
        # Add a tab from the bottom of the neighbor piece
        piece[:top, left+(tab_width // 2):right-(tab_width //2)] = neighbor_piece[:tab_height, (tab_width // 2):w-(tab_width//2)]
    
    elif direction == 'right' and action == -1:
        piece[top+(tab_height // 2):bottom-(tab_height//2), right:] = 0
    elif direction == 'bottom' and action == -1:
        piece[bottom:, left+(tab_width // 2):right-(tab_width //2)] = 0
    elif direction == 'left' and action == -1:
        piece[top+(tab_height // 2):bottom-(tab_height//2), :left] = 0
    elif direction == 'top' and action == -1:
        piece[:top, left+(tab_width // 2):right-(tab_width //2)] = 0

    return piece

def apply_matrix_to_pieces(pieces, puzzle_matrix):
    rows, cols, _ = puzzle_matrix.shape
    for i in range(rows):
        for j in range(cols):
            piece = pieces[i][j]
            left, top, right, bottom = puzzle_matrix[i, j]

            h, w, _ = piece.shape
            tab_width = w // 5
            tab_height = h // 5
            measurments = [tab_height, tab_width, h, w]

            resize_piece = np.full((h + 2 * tab_height, w + 2 * tab_width, 3), (0, 255, 0), dtype=np.uint8)
            resize_piece[tab_height:tab_height+h, tab_width:tab_width+w ] = piece

            if right != 0 and j < cols - 1:  
                pieces[i][j] = modify_piece_with_tabs(resize_piece, measurments, 'right', pieces[i][j + 1], action=right)
            if bottom != 0 and i < rows - 1:  
                pieces[i][j] = modify_piece_with_tabs(resize_piece, measurments, 'bottom', pieces[i + 1][j], action=bottom)
            if left != 0 and j > 0:  
                pieces[i][j] = modify_piece_with_tabs(resize_piece, measurments, 'left', pieces[i][j - 1], action=left)
            if top != 0 and i > 0:  
                pieces[i][j] = modify_piece_with_tabs(resize_piece, measurments, 'top', pieces[i - 1][j], action=top)

    return pieces

def display_puzzle(pieces):
    """Display the puzzle pieces using matplotlib."""
    rows = len(pieces)
    cols = len(pieces[0])

    # Assemble the image back from the puzzle pieces
    assembled_image = np.vstack([np.hstack(row) for row in pieces])

    # Show the assembled image
    plt.imshow(assembled_image)
    plt.axis('off')
    plt.show()

def main(image_path, rows, cols):
    # Step 1: Generate puzzle matrix
    puzzle_matrix = generate_puzzle_matrix(rows, cols)
    #print("Puzzle Matrix:")
    #print(puzzle_matrix)

    # Step 2: Split the image into pieces
    pieces = split_image(image_path, rows, cols)

    # Step 3: Modify pieces based on the puzzle matrix
    pieces = apply_matrix_to_pieces(pieces, puzzle_matrix)

    # Step 4: Display the puzzle pieces
    display_puzzle(pieces)

# Example usage
image_path = 'cat.jpg'  # Replace with your image path
rows, cols = 4, 4  # Number of rows and columns in the puzzle
main(image_path, rows, cols)