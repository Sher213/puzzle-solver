package src.main.com.example.app;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.dnd.*;
import java.awt.datatransfer.*;
import java.awt.image.BufferedImage;
import java.io.Console;
import java.io.File;
import java.util.ArrayList;

public class JigsawPuzzleGame {
    // Constants
    private static final int GRID_SIZE = 4; // 4x4 grid
    private JFrame frame;
    private JPanel previewPanel, puzzlePanel, holdingAreaPanel;
    private JFileChooser fileChooser;
    private ImageIcon imageIcon;
    private JButton[][] puzzlePiecesButtons;
    private BufferedImage[][] puzzlePieces;
    private ArrayList<JButton> holdingAreaButtons;
    
    // JLabel to display the mouse position
    private JLabel mousePositionLabel;

    public JigsawPuzzleGame() {
        // Create the main frame
        frame = new JFrame("Jigsaw Puzzle Game");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new BorderLayout());

        // Set the frame to a near-fullscreen size
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
        int width = (int) (screenSize.width * 0.6);
        int height = (int) (screenSize.height * 0.6);
        frame.setSize(width, height);
        frame.setLocationRelativeTo(null);

        // Create a menu bar
        JMenuBar menuBar = new JMenuBar();
        JMenu fileMenu = new JMenu("File");
        JMenuItem uploadItem = new JMenuItem("Upload Image");

        fileMenu.add(uploadItem);
        menuBar.add(fileMenu);
        frame.setJMenuBar(menuBar);

        // Create a mouse position label
        mousePositionLabel = new JLabel("Mouse Position: (0, 0)", SwingConstants.CENTER);
        mousePositionLabel.setPreferredSize(new Dimension(frame.getWidth(), 30));
        frame.add(mousePositionLabel, BorderLayout.NORTH);

        // Create a preview panel
        previewPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                if (imageIcon != null) {
                    int panelWidth = getWidth();
                    int panelHeight = getHeight();

                    int imageWidth = imageIcon.getIconWidth();
                    int imageHeight = imageIcon.getIconHeight();

                    // Calculate the aspect ratio
                    double imageAspect = (double) imageWidth / imageHeight;
                    double panelAspect = (double) panelWidth / panelHeight;

                    int drawWidth, drawHeight;
                    if (imageAspect > panelAspect) {
                        drawWidth = panelWidth;
                        drawHeight = (int) (panelWidth / imageAspect);
                    } else {
                        drawHeight = panelHeight;
                        drawWidth = (int) (panelHeight * imageAspect);
                    }

                    int x = (panelWidth - drawWidth) / 2;
                    int y = (panelHeight - drawHeight) / 2;

                    g.drawImage(imageIcon.getImage(), x, y, drawWidth, drawHeight, this);
                }
            }
        };
        previewPanel.setPreferredSize(new Dimension(200, 600));
        previewPanel.setBorder(BorderFactory.createTitledBorder("Image Preview"));
        frame.add(previewPanel, BorderLayout.WEST);

        // Create a puzzle panel
        puzzlePanel = new JPanel();
        puzzlePanel.setLayout(new GridLayout(GRID_SIZE, GRID_SIZE));
        puzzlePanel.setBorder(BorderFactory.createTitledBorder("Puzzle"));
        frame.add(puzzlePanel, BorderLayout.CENTER);

        // Create a holding area panel (right side)
        holdingAreaPanel = new JPanel();
        holdingAreaPanel.setLayout(new FlowLayout(FlowLayout.LEFT, 5, 5));
        holdingAreaPanel.setPreferredSize(new Dimension(200, height));
        holdingAreaPanel.setBorder(BorderFactory.createTitledBorder("Holding Area"));
        frame.add(holdingAreaPanel, BorderLayout.EAST);

        // Initialize the holding area list
        holdingAreaButtons = new ArrayList<>();

        // File chooser for uploading images
        fileChooser = new JFileChooser();

        // Add action listener for "Upload Image"
        uploadItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                int returnValue = fileChooser.showOpenDialog(frame);
                if (returnValue == JFileChooser.APPROVE_OPTION) {
                    File selectedFile = fileChooser.getSelectedFile();
                    displayImage(selectedFile);
                }
            }
        });

        // Mouse motion listener to update the mouse position
        frame.addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseMoved(MouseEvent e) {
                // Update the mouse position label
                mousePositionLabel.setText("Mouse Position: (" + e.getX() + ", " + e.getY() + ")");
            }
        });

        // Show the frame
        frame.setVisible(true);
    }

    // Method to display the uploaded image in the preview panel
    private void displayImage(File imageFile) {
        imageIcon = new ImageIcon(imageFile.getAbsolutePath());

        int imageWidth = imageIcon.getIconWidth();
        int imageHeight = imageIcon.getIconHeight();

        previewPanel.repaint(); // Trigger re-painting of the preview panel
        generatePuzzleGrid(imageWidth, imageHeight);
    }

    // Method to generate a 4x4 puzzle grid with draggable pieces
    private void generatePuzzleGrid(int imageWidth, int imageHeight) {
        puzzlePanel.removeAll(); // Clear the puzzle panel
        holdingAreaPanel.removeAll(); // Clear the holding area
        holdingAreaButtons.clear(); // Clear the holding area buttons list

        int pieceWidth = imageWidth / GRID_SIZE;
        int pieceHeight = imageHeight / GRID_SIZE;

        // Convert ImageIcon to BufferedImage for subimage slicing
        BufferedImage bufferedImage = new BufferedImage(imageWidth, imageHeight, BufferedImage.TYPE_INT_ARGB);
        Graphics g = bufferedImage.createGraphics();
        g.drawImage(imageIcon.getImage(), 0, 0, null);
        g.dispose();

        // Initialize the puzzle pieces array
        puzzlePieces = new BufferedImage[GRID_SIZE][GRID_SIZE];
        puzzlePiecesButtons = new JButton[GRID_SIZE][GRID_SIZE];

        // Slice the image into puzzle pieces and create buttons for each piece
        for (int row = 0; row < GRID_SIZE; row++) {
            for (int col = 0; col < GRID_SIZE; col++) {
                puzzlePieces[row][col] = bufferedImage.getSubimage(col * pieceWidth, row * pieceHeight, pieceWidth, pieceHeight);

                // Create a JButton for each piece
                JButton pieceButton = createDraggablePuzzlePiece(puzzlePieces[row][col], row, col);

                // Add the piece to the puzzle grid
                puzzlePanel.add(pieceButton);
                puzzlePiecesButtons[row][col] = pieceButton;
            }
        }
        puzzlePanel.revalidate();
        puzzlePanel.repaint();
    }

    private JButton createDraggablePuzzlePiece(BufferedImage pieceImage, int row, int col) {
        JButton pieceButton = new JButton(new ImageIcon(pieceImage));
        pieceButton.setPreferredSize(new Dimension(pieceImage.getWidth(), pieceImage.getHeight()));
        pieceButton.setBorder(null);
    
        // MouseListener to handle the press and release
        pieceButton.addMouseListener(new MouseAdapter() {
            public void mousePressed(MouseEvent e) {
                // Record the starting point of the drag
                pieceButton.setCursor(Cursor.getPredefinedCursor(Cursor.MOVE_CURSOR));
            }
    
            public void mouseReleased(MouseEvent e) {
                pieceButton.setCursor(Cursor.getPredefinedCursor(Cursor.DEFAULT_CURSOR));
                // Handle dropping behavior
                handleDrop(pieceButton, e, row, col);
            }
        });
    
        // MouseMotionListener to update the position of the piece as it's dragged
        pieceButton.addMouseMotionListener(new MouseMotionAdapter() {
            private Point offset = null;
    
            @Override
            public void mouseDragged(MouseEvent e) {
                if (offset == null) {
                    // Calculate the offset from the cursor to the top-left corner of the button
                    offset = new Point(500, 250);
                }
    
                // Move the piece along with the mouse
                int newX = e.getXOnScreen() - offset.x;
                int newY = e.getYOnScreen() - offset.y;
    
                pieceButton.setLocation(newX, newY);  // Update the position of the button
            }
        });
    
        return pieceButton;
    }

    private void handleDrop(JButton pieceButton, MouseEvent e, int originalRow, int originalCol) {
        // Define behavior for when the piece is dropped
        Container parent = pieceButton.getParent();
        Point dropLocation = e.getLocationOnScreen();
        Component dropTarget = SwingUtilities.getDeepestComponentAt(
            SwingUtilities.getRootPane(pieceButton),
            dropLocation.x,
            dropLocation.y
        );

        if (dropTarget instanceof JButton) {
            JButton targetButton = (JButton) dropTarget;

            // Swap the positions between the current and target pieces
            swapPieces(pieceButton, targetButton, originalRow, originalCol);
        }
    }

    private void swapPieces(JButton pieceButton, JButton targetButton, int originalRow, int originalCol) {
        // Get the position of the target piece
        Point targetPosition = targetButton.getLocation();
        int targetRow = targetPosition.y / pieceButton.getHeight();
        int targetCol = targetPosition.x / pieceButton.getWidth();

        // Check if the drop target is valid
        if (targetRow >= 0 && targetRow < GRID_SIZE && targetCol >= 0 && targetCol < GRID_SIZE && pieceButton != null) {
            // Swap the images between the two pieces
            ImageIcon tempIcon = (ImageIcon) puzzlePiecesButtons[originalRow][originalCol].getIcon();
            puzzlePiecesButtons[originalRow][originalCol].setIcon((ImageIcon) targetButton.getIcon());
            targetButton.setIcon(tempIcon);

            puzzlePiecesButtons[originalRow][originalCol] = targetButton;
            puzzlePiecesButtons[targetRow][targetCol] = pieceButton;

            puzzlePanel.remove(puzzlePiecesButtons[originalRow][originalCol]);
            puzzlePanel.add(puzzlePiecesButtons[originalRow][originalCol], targetRow * GRID_SIZE + targetCol);
            puzzlePanel.remove(puzzlePiecesButtons[targetRow][targetCol]);
            puzzlePanel.add(puzzlePiecesButtons[targetRow][targetCol], originalRow * GRID_SIZE + originalCol);
            // Revalidate and repaint the puzzle panel to reflect the changes
            puzzlePanel.revalidate();
            puzzlePanel.repaint();
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                new JigsawPuzzleGame();
            }
        });
    }
}
