package src.main.com.example.utilities;
import javax.swing.ImageIcon;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class PuzzlePiece {
    
    public BufferedImage image;
    public JButton button;
    public int row;
    public int col;

    public PuzzlePiece(BufferedImage pieceImage, int row, int col) {
        
        this.button = new JButton(new ImageIcon(pieceImage));
        button.setPreferredSize(new Dimension(pieceImage.getWidth(), pieceImage.getHeight()));
        button.setBorder(null);
        
        this.image = pieceImage;
        this.row = row;
        this.col = col;
    }
}
