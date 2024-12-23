import os
import time
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def combine_csvs():
    # Directory containing CSV files
    directory = './'

    # List to store individual DataFrames
    dataframes = []

    # Loop through all files in the directory
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            dataframes.append(df)

    # Concatenate all DataFrames
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv('combined_output.csv', index=False)

    print("CSV files combined successfully!")

def number_shuffled_pieces(assembled_img_path, shuffled_img_path, row_col, piece_w, piece_h):
    df_out = pd.DataFrame(columns=["row_num", "col_num", "position_in_puzzle", "row_col"])

    # Open the image files
    with Image.open(assembled_img_path) as a_img, Image.open(shuffled_img_path) as s_img:
        # Ensure both images have the same size
        assert a_img.size == s_img.size, "Images must have the same dimensions"
        
        # Get width and height of the images
        width, height = a_img.size
        
        # Create drawing objects to add text
        draw_a = ImageDraw.Draw(a_img)
        draw_s = ImageDraw.Draw(s_img)

        # Optionally, load a font
        try:
            font = ImageFont.truetype("arial.ttf", piece_w // 3)  # Adjust font size as needed
        except IOError:
            font = ImageFont.load_default()

        piece_num = 0 # To track the numbering of pieces
        matched_pieces = set()  # To keep track of matched shuffled pieces

        for row in range(row_col):
            for col in range(row_col):
                # Calculate coordinates for the piece
                left = col * piece_w
                top = row * piece_h
                right = left + piece_w
                bottom = top + piece_h

                # Extract the piece from assembled
                assembled_piece = a_img.crop((left, top, right, bottom))
                
                # Convert piece to numpy arrays for comparison
                assembled_array = np.array(assembled_piece)

                for row2 in range(row_col):
                    for col2 in range(row_col):
                        # Calculate coordinates for the piece
                        left2 = col2 * piece_w
                        top2 = row2 * piece_h
                        right2 = left2 + piece_w
                        bottom2 = top2 + piece_h
                        
                        # Check if this piece has already been matched
                        if (row2, col2) in matched_pieces:
                            continue

                        #loop through shuffled to find the matching piece
                        shuffled_piece = s_img.crop((left2, top2, right2, bottom2))
                        shuffled_array = np.array(shuffled_piece)

                        # Check if pieces match (element-wise comparison)
                        if np.array_equal(assembled_array, shuffled_array):
                            piece_num += 1

                            # Mark this shuffled piece as matched
                            matched_pieces.add((row2, col2))

                            # Draw numbers on the assembled image
                            piece_center_x = (left + right) // 2
                            piece_center_y = (top + bottom) // 2
                            draw_a.text((piece_center_x - 10, piece_center_y - 10), str(piece_num), fill="red", font=font)

                            # Draw numbers on the shuffled image
                            piece_center_x2 = (left2 + right2) // 2
                            piece_center_y2 = (top2 + bottom2) // 2
                            draw_s.text((piece_center_x2 - 10, piece_center_y2 - 10), str(piece_num), fill="blue", font=font)

                            new_entry = pd.DataFrame([{
                                "row_num": row, 
                                "col_num": col,
                                "position_in_puzzle": piece_num,
                                "row_col": row_col,
                                "position_normalized": piece_num / (row_col**2)}])
                            
                            df_out = pd.concat([df_out, new_entry], ignore_index=True)
                            break
                    else:
                        continue
                    break

        assembled_save_path = os.path.join("./assembled-numbered", f"{assembled_img_path.split(chr(92))[-1].split('.')[0]}-numbered.png")
        shuffled_save_path = os.path.join("./shuffled-numbered", f"{shuffled_img_path.split(chr(92))[-1].split('.')[0]}-numbered.png")
        
        #a_img.imshow()
        #s_img.imshow()
        a_img.save(assembled_save_path)
        s_img.save(shuffled_save_path)
        df_out.to_csv(f"./puzzles-numbered-csv/{assembled_img_path.split(chr(92))[-1].split('.')[0]}_puzzle_numbered.csv")

def gen_puzzles_csv():
    root_path = "../puzzle-maker/"

    #name,original_image,assembled_puzzle,shuffled_puzzle,puzzle_row_col,single_piece_width,single_piece_height
    df = pd.read_csv("../puzzle-maker/images-and-puzzles.csv")

    # Filter out unwanted directories
    dirs = [
        dir for dir in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, dir)) and 
        not ("cat.jpg" in dir or ".csv" in dir or ".py" in dir)
    ]

    # Ensure there are at least two directories to process
    if len(dirs) < 2:
        raise ValueError("Not enough directories to process. At least two are required.")

    # Access two directories at once
    dir1, dir2 = dirs[:2]  # Use the first two directories (you can change this logic)

    dir1_path = os.path.join(root_path, dir1)
    dir2_path = os.path.join(root_path, dir2)

    # Ensure both directories exist
    assert os.path.exists(dir1_path), f"Directory {dir1_path} does not exist."
    assert os.path.exists(dir2_path), f"Directory {dir2_path} does not exist."

    # Get files from both directories
    files_dir1 = sorted(os.listdir(dir1_path))
    files_dir2 = sorted(os.listdir(dir2_path))

    # Iterate through files from both directories simultaneously
    for file_index, (file1, file2) in enumerate(zip(files_dir1, files_dir2)):
        file1_path = os.path.join(dir1_path, file1)
        file2_path = os.path.join(dir2_path, file2)

        print(f"Processing pair {file_index}:")
        print(f"  File from {dir1}: {file1_path}")
        print(f"  File from {dir2}: {file2_path}")

        file_inst = df[df['name'] == file1.split('-')[0]].index
        row_col = df.iloc[file_inst]['puzzle_row_col'].values[0]
        piece_w = df.iloc[file_inst]['single_piece_width'].values[0]
        piece_h = df.iloc[file_inst]['single_piece_height'].values[0]

        number_shuffled_pieces(
            assembled_img_path=file1_path,
            shuffled_img_path=file2_path,
            row_col=row_col,
            piece_w=piece_w,
            piece_h=piece_h
        )

    combine_csvs()
