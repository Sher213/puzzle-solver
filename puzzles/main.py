from PIL import Image, ImageDraw, ImageOps
import numpy as np
import random
import matplotlib.pyplot as plt

def create_jigsaw_puzzle(image_path, num_pieces=50):
    # Load the image
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Calculate number of rows and columns
    num_rows = int((num_pieces ** 0.5) + 0.5)
    num_cols = num_rows
    piece_width = img_width // num_cols
    piece_height = img_height // num_rows
    
    # Create puzzle piece masks
    def create_piece_mask():
        mask = Image.new("L", (piece_width, piece_height), 0)
        draw = ImageDraw.Draw(mask)
        
        # Draw the piece rectangle with random tabs/blanks
        def draw_tab_or_blank(x, y, is_vertical):
            size = piece_width // 5  # Size of the tab/blank
            tab_direction = random.choice([-1, 1])  # Randomly choose a tab (+1) or blank (-1)
            if is_vertical:
                # Create a vertical tab or blank
                y0 = y - size * (1 if tab_direction == -1 else 0)  # Top edge
                y1 = y + size * (1 if tab_direction == 1 else 0)  # Bottom edge
                draw.ellipse((x - size, y0, x + size, y1), fill=255)
            else:
                # Create a horizontal tab or blank
                x0 = x - size * (1 if tab_direction == -1 else 0)  # Left edge
                x1 = x + size * (1 if tab_direction == 1 else 0)  # Right edge
                draw.ellipse((x0, y - size, x1, y + size), fill=255)
        
        # Ensure at least one tab or blank for top, bottom, left, right 
        draw.rectangle((0, 0, piece_width, piece_height), fill=255) 
        draw_tab_or_blank(piece_width // 2, 0, False) # Top 
        draw_tab_or_blank(piece_width // 2, piece_height - piece_width // 5, False) # Bottom 
        draw_tab_or_blank(0, piece_height // 2, True) # Left 
        draw_tab_or_blank(piece_width - piece_width // 5, piece_height // 2, True) #Right
        
        return mask

    # Generate pieces
    pieces = []
    for row in range(num_rows):
        for col in range(num_cols):
            left = col * piece_width
            top = row * piece_height
            right = left + piece_width
            bottom = top + piece_height
            piece = img.crop((left, top, right, bottom))
            mask = create_piece_mask()
            piece = Image.composite(piece, Image.new("RGB", piece.size, "white"), mask)
            pieces.append((piece, (row, col), mask))
    
    # Shuffle pieces
    random.shuffle(pieces)
    
    # Create a blank canvas to place pieces
    canvas = Image.new("RGB", (img_width, img_height), "white")
    for index, (piece, (orig_row, orig_col), mask) in enumerate(pieces):
        new_row = index // num_cols
        new_col = index % num_cols
        top = new_row * piece_height
        left = new_col * piece_width
        canvas.paste(piece, (left, top), mask)
    
    # Show the shuffled puzzle
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(canvas)
    plt.title("Randomized Jigsaw Puzzle with Piece Shapes")
    plt.show()

# Example usage
image_path = 'D:\coding-work\puzzle-solver\puzzles\Download-From-Google-Images\cats high quality images\cats high quality images30.jpeg'  # Replace with the path to your image
create_jigsaw_puzzle(image_path)